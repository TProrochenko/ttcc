.PHONY: test
test:
	gcc -O3 -o model web/model.c -lm \
	&& ./model \
	&& rm model

.PHONY: web
web:
	emcc -O3 web/model.c \
	-s EXPORTED_FUNCTIONS='["_init", "_complete"]' \
	-s EXPORTED_RUNTIME_METHODS='["cwrap", "FS"]' \
    -o web/model.js \
    -s ALLOW_MEMORY_GROWTH=1 
