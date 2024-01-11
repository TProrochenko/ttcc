.PHONY: test
test:
	gcc -O3 -o model web/model.c -lm \
	&& ./model \
	&& rm model

.PHONY: web
web:
	emcc -O3 -flto -msimd128 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s ASSERTIONS=0 \
	-s EXPORTED_FUNCTIONS='["_init", "_complete"]' \
	-s EXPORTED_RUNTIME_METHODS='["cwrap", "FS"]' \
	web/model.c -o web/model.js
