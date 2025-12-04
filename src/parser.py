from __future__ import annotations

from src.lexer import Token, TokenType
import src.st as st


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

        self._furthest = -1
        self._errors = []

    def parse(self) -> st.Program | None:
        start = self._mark()

        definitions = []
        while True:
            pos = self._mark()
            if defn := self.parse_definition():
                definitions.append(defn)
                continue

            self._reset(pos)
            if self._expect_eof():
                return st.Program(definitions)

            self._reset(start)
            return None

    def print_errors(self) -> None:
        while len(self._errors) > 0:
            loc, msg = self._errors.pop()
            print(f"{loc} {msg}")

    def parse_definition(self) -> st.Definition | None:
        start = self._mark()
        if (self._expect_keyword("def")
                and (name := self._expect_word())
                and self._expect_keyword("is")
                and (body := self.parse_expression())
                and self._expect_punc(";")):
            return st.WordDef(name.name, body)

        self._error_at(start, f"Error: Could not parse Word definition.")
        self._reset(start)
        return None

    def parse_statement(self) -> st.Statement | None:
        start = self._mark()
        if if_stmt := self.parse_if():
            return if_stmt

        self._reset(start)
        if while_stmt := self.parse_while():
            return while_stmt

        self._reset(start)
        if local_stmt := self.parse_locals():
            return local_stmt

        self._reset(start)
        return None

    def parse_if(self) -> st.If | None:
        start = self._mark()
        if (self._expect_keyword("if")
                and (cond := self.parse_expression())
                and self._expect_keyword("then")
                and (body := self.parse_expression())):

            if_stmt = st.If(cond, body, None)

            pos = self._mark()
            add_els_to = if_stmt
            while (self._expect_keyword("else-if")
                   and (el_cond := self.parse_expression())
                   and self._expect_keyword("then")
                   and (el_body := self.parse_expression())):
                pos = self._mark()

                els = st.If(el_cond, el_body, None)
                add_els_to.els = st.Expression([els])
                add_els_to = els

            self._reset(pos)
            if self._expect_keyword("else") and (el_body := self.parse_expression()):
                add_els_to.els = el_body

            if self._expect_punc(";"):
                return if_stmt

        self._error_at(start, f"Error: Could not parse if statement.")
        self._reset(start)
        return None

    def parse_while(self) -> st.While | None:
        start = self._mark()
        if (self._expect_keyword("while")
                and (cond := self.parse_expression())
                and self._expect_keyword("do")
                and (body := self.parse_expression())
                and self._expect_punc(";")):
            return st.While(cond, body)

        self._error_at(start, f"Error: Could not parse while loop.")
        self._reset(start)
        return None

    def parse_locals(self) -> st.Locals | None:
        start = self._mark()

        if self._expect_keyword("->") and (name := self._expect_word()):
            names = [name.name]

            pos = self._mark()
            while self._expect_punc(",") and (name := self._expect_word()):
                names.append(name.name)
                pos = self._mark()

            self._reset(pos)
            if self._expect_keyword("do") and (body := self.parse_expression()) and self._expect_punc(";"):
                return st.Locals(names, body)

        self._error_at(start, f"Error: Could not parse local binding.")
        self._reset(start)
        return None

    def parse_expression(self) -> st.Expression | None:
        words = []
        while True:
            pos = self._mark()
            if word := self._expect_word():
                words.append(word)
                continue

            self._reset(pos)
            if num := self._expect_number():
                words.append(num)
                continue

            self._reset(pos)
            if stmt := self.parse_statement():
                words.append(stmt)
                continue

            return st.Expression(words)

    def _expect_keyword(self, lexeme: str) -> bool:
        start = self._mark()
        if not self._has_more():
            self._error_at(-1, f"Error: Expected KEYWORD '{lexeme}' but got nothing instead.")
            return False

        current = self._advance()
        if current.typ == TokenType.KEYWORD and current.lexeme == lexeme:
            return True

        self._error_at(start, f"Error: Expected KEYWORD '{lexeme}' but got {current.typ.name} '{current.lexeme}' instead.")
        self._reset(start)
        return False

    def _expect_word(self) -> st.Word | None:
        start = self._mark()
        if not self._has_more():
            self._error_at(-1, f"Error: Expected WORD but got nothing instead.")
            return None

        current = self._advance()
        if not current.typ == TokenType.WORD:
            self._error_at(start, f"Error: Expected WORD but got {current.typ.name} '{current.lexeme}' instead.")
            self._reset(start)
            return None

        return st.Word(current.lexeme)

    def _expect_number(self) -> st.IntLiteral | None:
        start = self._mark()
        if not self._has_more():
            self._error_at(-1, f"Error: Expected NUMBER but got nothing instead.")
            return None

        current = self._advance()
        if not current.typ == TokenType.NUMBER:
            self._error_at(start, f"Error: Expected NUMBER but got {current.typ.name} '{current.lexeme}' instead.")
            self._reset(start)
            return None

        return st.IntLiteral(int(current.lexeme))

    def _expect_punc(self, lexeme: str) -> bool:
        start = self._mark()
        if not self._has_more():
            self._error_at(-1, f"Error: Expected PUNCTUATION '{lexeme}' but got nothing instead.")
            return False

        current = self._advance()
        if current.typ == TokenType.PUNCTUATION and current.lexeme == lexeme:
            return True

        self._error_at(start, f"Error: Expected PUNCTUATION '{lexeme}' but got {current.typ.name} '{current.lexeme}' instead.")
        self._reset(start)
        return False

    def _expect_eof(self) -> bool:
        start = self._mark()
        if not self._has_more():
            self._error_at(-1, f"Error: File ended unexpectedly.")
            return False

        current = self._advance()
        if current.typ == TokenType.EOF:
            return True

        self._reset(start)
        return False

    def _has_more(self) -> bool:
        return self._pos < len(self._tokens)

    def _current(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        current = self._current()
        self._pos += 1
        return current

    def _mark(self) -> int:
        if self._pos >= self._furthest:
            self._furthest = self._pos
            self._errors = []

        return self._pos

    def _reset(self, mark: int) -> None:
        self._pos = mark

    def _error_at(self, mark: int, message: str) -> None:
        if not mark < len(self._tokens):
            last_token = self._tokens[-1]
            loc = last_token.loc
            loc.col += len(last_token.lexeme)
        else:
            loc = self._tokens[mark].loc
        self._errors.append((loc, message))
