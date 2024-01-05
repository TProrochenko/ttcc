#include <stdio.h>
#include <stdlib.h>
// #include <ctype.h>
#include <stdint.h>
// #include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

// ----------------------------------------------------------------------------
// Globals
int GS = 0;

// ----------------------------------------------------------------------------
// Transformer model

typedef struct
{
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct
{
    int8_t *q;
    float *s;
} QuantizedTensor;

typedef struct
{
    QuantizedTensor *q_tokens;
    float *token_embedding_table;

    float *rms_att_weight;
    float *rms_ffn_weight;

    QuantizedTensor *wq;
    QuantizedTensor *wk;
    QuantizedTensor *wv;
    QuantizedTensor *wo;

    QuantizedTensor *w1;
    QuantizedTensor *w2;
    QuantizedTensor *w3;

    float *rms_final_weight;

    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct
{
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    QuantizedTensor xq;
    QuantizedTensor hq;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

typedef struct
{
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float *data;
    ssize_t file_size;
} Transformer;

void malloc_run_state(RunState *s, Config *p)
{
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor){.q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim, sizeof(float))};
    s->hq = (QuantizedTensor){.q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim, sizeof(float))};
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float *x, int n)
{
    for (int i = 0; i < n; i++)
    {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float *x, int n)
{
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++)
    {

        float wmax = 0.0;
        for (int i = 0; i < GS; i++)
        {
            float val = fabs(x[group * GS + i]);
            if (val > wmax)
            {
                wmax = val;
            }
        }

        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        for (int i = 0; i < GS; i++)
        {
            float quant_value = x[group * GS + i] / scale;
            int8_t quantized = (int8_t)round(quant_value);
            qx->q[group * GS + i] = quantized;
        }
    }
}

QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each)
{
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for (int i = 0; i < n; i++)
    {
        res[i].q = (int8_t *)p;
        p = (int8_t *)p + size_each;
        res[i].s = (float *)p;
        p = (float *)p + size_each / GS;
    }
    *ptr = p;
    return res;
}

void memory_map_weights(TransformerWeights *w, Config *p, void *ptr, uint8_t shared_classifier)
{
    int head_size = p->dim / p->n_heads;
    float *fptr = (float *)ptr;
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    ptr = (void *)fptr;
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(const char *checkpoint, Config *config, TransformerWeights *weights,
                     int *fd, float **data, ssize_t *file_size)
{
    FILE *file = fopen(checkpoint, "rb");
    if (!file)
    {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1)
    {
        exit(EXIT_FAILURE);
    }
    if (magic_number != 0x616b3432)
    {
        fprintf(stderr, "Bad magic number\n");
        exit(EXIT_FAILURE);
    }
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1)
    {
        exit(EXIT_FAILURE);
    }
    if (version != 2)
    {
        fprintf(stderr, "Bad version %d, need version 2\n", version);
        exit(EXIT_FAILURE);
    }
    int header_size = 256; 
    if (fread(config, sizeof(Config), 1, file) != 1)
    {
        exit(EXIT_FAILURE);
    }

    uint8_t shared_classifier; 
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1)
    {
        exit(EXIT_FAILURE);
    }
    int group_size; 
    if (fread(&group_size, sizeof(int), 1, file) != 1)
    {
        exit(EXIT_FAILURE);
    }
    GS = group_size;
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1)
    {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED)
    {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    void *weights_ptr = ((char *)*data) + header_size;
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

void build_transformer(Transformer *t, const char *checkpoint_path)
{
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

// ----------------------------------------------------------------------------

void rmsnorm(float *o, float *x, float *weight, int size)
{
    float ss = 0.0f;
    for (int j = 0; j < size; j++)
    {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++)
    {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float *x, int size)
{
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

void matmul(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d)
{

    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++)
    {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        int j;
        for (j = 0; j <= n - GS; j += GS)
        {
            for (int k = 0; k < GS; k++)
            {
                ival += ((int32_t)x->q[j + k]) * ((int32_t)w->q[in + j + k]);
            }
            val += ((float)ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
}

float *forward(Transformer *transformer, int token, int pos)
{

    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++)
    {

        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        for (int i = 0; i < dim; i += 2)
        {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++)
            {
                float *vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        int loff = l * p->seq_len * kv_dim;
        float *key_cache_row = s->key_cache + loff + pos * kv_dim;
        float *value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++)
        {
            float *q = s->q + h * head_size;
            float *att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++)
            {
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++)
                {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++)
            {
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++)
                {
                    xb[i] += a * v[i];
                }
            }
        }

        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb2[i];
        }

        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; i++)
        {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb[i];
        }
    }

    rmsnorm(x, x, w->rms_final_weight, dim);

    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------

void encode(char *str, int *tokens, int seq_len)
{
    int pos = 0;
    tokens[pos++] = 2;

    while (*str && pos < seq_len)
    {
        tokens[pos++] = *str++;
    }

    memset(tokens + pos, 0, (seq_len - pos) * sizeof(int));
}

void decode(int *tokens, char *str, int seq_len)
{
    int pos = 0;
    while (pos < seq_len && tokens[pos + 1] != 0)
    {
        str[pos] = (char)tokens[pos + 1];
        pos++;
    }
    str[pos] = '\0';
}

// ----------------------------------------------------------------------------

Transformer transformer;
char *completion = NULL;

char *complete(char *code)
{
    int seq_len = transformer.config.seq_len;
    int vocab_size = transformer.config.vocab_size;
    int tokens[seq_len];
    float cprob = 1.;
    int code_len = strlen(code);

    encode(code, tokens, seq_len);

    int pos = 0;
    for (int i = 0; i < code_len; i++)
    {
        if (code[i] != completion[i])
        {
            break;
        }
        pos++;
    }

    while (pos < seq_len)
    {
        float *logits = forward(&transformer, tokens[pos], pos);
        softmax(logits, vocab_size);

        if (pos >= code_len)
        {
            int token = 0;
            float max_p = logits[0];
            for (int i = 1; i < vocab_size; i++)
            {
                if (logits[i] > max_p)
                {
                    token = i;
                    max_p = logits[i];
                }
            }

            cprob *= max_p;
            if (token == 3 | cprob < 0.5)
            {
                break;
            }
            tokens[pos + 1] = token;
        }

        pos++;
    }
    decode(tokens, completion, seq_len);

    return completion;
}

// ----------------------------------------------------------------------------

void init(const char *filename)
{
    build_transformer(&transformer, filename);
    completion = (char *)malloc(transformer.config.seq_len * sizeof(char));
}

#ifndef __EMSCRIPTEN__
int main()
{
    init("models/python-lemon-violet.bin");
    complete("import to");
    printf("%s\n", completion);

    init("models/javascript-divine-blaze.bin");
    complete("document.ad");
    printf("%s\n", completion);

    return 0;
}
#endif
