document.addEventListener("DOMContentLoaded", () => {
  const editor = new Editor();
  editor.init();
});

class Editor {
  constructor() {
    this.input = document.getElementById("input-area");
    this.hint = document.getElementById("hint-area");
    this.timeout = null;
  }

  init() {
    this.input.addEventListener("input", () => this.debounceInput());
    this.input.addEventListener("click", () => this.clearMirrorArea());
    this.input.addEventListener("keydown", (event) =>
      this.handleKeyDown(event)
    );
  }

  debounceInput() {
    clearTimeout(this.timeout);
    this.timeout = setTimeout(() => this.handleInput(), 50);
  }

  handleInput() {
    if (this.isCaretAtEnd()) {
      this.getAutocompleteSuggestion();
    } else {
      this.trimAfterCaret();
    }
  }

  clearMirrorArea() {
    if (!this.isCaretAtEnd()) {
      this.hint.value = "";
    }
  }

  handleKeyDown(event) {
    switch (event.key) {
      case "Tab":
        event.preventDefault();
        this.hint.value ? this.applySuggestion() : this.insertTabAtCaret();
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

  getAutocompleteSuggestion() {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:5000/autocomplete", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onload = () => {
      if (xhr.status === 200) {
        const response = JSON.parse(xhr.responseText);
        this.hint.value = this.input.value.replace(/./g, " ") + response;
      }
    };
    xhr.send(JSON.stringify(this.input.value));
  }

  insertTabAtCaret() {
    const caretPosition = this.input.selectionStart;
    const beforeCaret = this.input.value.substring(0, caretPosition);
    const afterCaret = this.input.value.substring(caretPosition);
    this.input.value = beforeCaret + "    " + afterCaret;
    this.input.selectionStart = this.input.selectionEnd = caretPosition + 4;
  }

  trimAfterCaret() {
    const caretPosition = this.input.selectionStart;
    const beforeCaret = this.input.value.substring(0, caretPosition);
    const afterCaret = this.input.value.substring(caretPosition);
    this.input.value = beforeCaret + afterCaret.trimEnd();
  }

  applySuggestion() {
    this.input.value += this.hint.value.slice(this.input.value.length);
    this.input.selectionStart = this.input.selectionEnd =
      this.input.value.length;
    this.hint.value = "";
    this.handleInput();
  }

  isCaretAtEnd() {
    return this.input.value.length === this.input.selectionStart;
  }
}
