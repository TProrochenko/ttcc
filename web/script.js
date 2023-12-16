class Editor {
  constructor() {
    this.code = document.getElementById("code-area");
    this.hint = document.getElementById("hint-area");
    this.lang = document.getElementById("language");
    this.loadingBar = document.getElementById("loading-bar");
    this.loadingProgress = document.getElementById("loading-progress");
    this.timeout = null;

    this.code.addEventListener("input", () => this.debounceInput());
    this.code.addEventListener("click", () => this.clearHintArea());
    this.code.addEventListener("keydown", (event) => this.handleKeyDown(event));
    this.lang.addEventListener("change", () => this.initModel(this.lang.value));

    Module.onRuntimeInitialized = () => {
      this.initModel(this.lang.value);
    };
  }

  async initModel(filename) {
    try {
      let received = 0;
      this.loadingProgress.value = 0;
      this.code.readOnly = true;
      this.loadingBar.style.display = "block";

      const response = await fetch("models/" + filename);
      const reader = response.body.getReader();
      const expected = response.headers.get("Content-Length");
      let uint8Array = new Uint8Array(expected);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        uint8Array.set(value, received);
        received += value.length;
        this.loadingProgress.value = received / expected;
      }

      Module.FS.writeFile(filename, uint8Array);
      Module.cwrap("init", null, ["string"])(filename);
    } catch (error) {
      console.error(error.message);
    }

    this.loadingBar.style.display = "none";
    this.resetInputs();
  }

  resetInputs() {
    this.code.readOnly = false;
    this.code.value = "";
    this.hint.value = "";
    this.code.focus();
  }

  debounceInput() {
    clearTimeout(this.timeout);
    this.timeout = setTimeout(() => this.handleInput(), 50);
  }

  handleInput() {
    if (this.isCaretAtEnd()) this.getSuggestion();
  }

  clearHintArea() {
    if (!this.isCaretAtEnd()) this.hint.value = "";
  }

  handleKeyDown(event) {
    switch (event.key) {
      case "Tab":
        event.preventDefault();
        this.hint.value.slice(this.code.value.length)
          ? this.applySuggestion()
          : this.insertTabAtCaret();
        break;
      case "ArrowLeft":
      case "ArrowRight":
      case "ArrowUp":
      case "ArrowDown":
      case "Escape":
        this.hint.value = "";
        break;
    }
  }

  getSuggestion() {
    const complete = Module.cwrap("complete", "string", ["string"]);
    var completion = complete(this.code.value).slice(this.code.value.length);
    this.hint.value = this.code.value.replace(/./g, " ") + completion;
  }

  insertTabAtCaret() {
    const caretPosition = this.code.selectionStart;
    const beforeCaret = this.code.value.substring(0, caretPosition);
    const afterCaret = this.code.value.substring(caretPosition);
    this.code.value = beforeCaret + "    " + afterCaret;
    this.code.selectionStart = this.code.selectionEnd = caretPosition + 4;
  }

  applySuggestion() {
    this.code.value += this.hint.value.slice(this.code.value.length);
    this.code.selectionStart = this.code.selectionEnd = this.code.value.length;
    this.hint.value = "";
    this.handleInput();
  }

  isCaretAtEnd() {
    return this.code.value.length === this.code.selectionStart;
  }
}

document.addEventListener("DOMContentLoaded", () => new Editor());
