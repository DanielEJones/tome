from unittest import TestCase, main

from src.lexer import Token, TokenType, Loc
from src.parser import Parser
import src.st as st


L = Loc("test", 0, 0)


class TestParseFunction(TestCase):
    def test_no_word_function(self) -> None:
        tree = Parser([
            Token(TokenType.KEYWORD, "def", L),
            Token(TokenType.WORD, "empty", L),
            Token(TokenType.KEYWORD, "is", L),
            Token(TokenType.PUNCTUATION, ";", L),
            Token(TokenType.EOF, "", L)]).parse()

        self.assertEqual(
            st.Program([st.WordDef("empty", st.Expression([]))]),
            tree
        )

    def test_single_word_function(self) -> None:
        tree = Parser([
            Token(TokenType.KEYWORD, "def", L),
            Token(TokenType.WORD, "add", L),
            Token(TokenType.KEYWORD, "is", L),
            Token(TokenType.WORD, "+", L),
            Token(TokenType.PUNCTUATION, ";", L),
            Token(TokenType.EOF, "", L)]).parse()

        self.assertEqual(
            st.Program([st.WordDef("add", st.Expression([st.Word("+")]))]),
            tree
        )

    def test_multi_word_function(self) -> None:
        tree = Parser([
            Token(TokenType.KEYWORD, "def", L),
            Token(TokenType.WORD, "square", L),
            Token(TokenType.KEYWORD, "is", L),
            Token(TokenType.WORD, "dup", L),
            Token(TokenType.WORD, "*", L),
            Token(TokenType.PUNCTUATION, ";", L),
            Token(TokenType.EOF, "", L)]).parse()

        self.assertEqual(
            st.Program([st.WordDef("square", st.Expression([st.Word("dup"), st.Word("*")]))]),
            tree
        )

    def test_number_in_function(self) -> None:
        tree = Parser([
            Token(TokenType.KEYWORD, "def", L),
            Token(TokenType.WORD, "two-times", L),
            Token(TokenType.KEYWORD, "is", L),
            Token(TokenType.NUMBER, "2", L),
            Token(TokenType.WORD, "*", L),
            Token(TokenType.PUNCTUATION, ";", L),
            Token(TokenType.EOF, "", L)]).parse()

        self.assertEqual(
            st.Program([st.WordDef("two-times", st.Expression([st.IntLiteral(2), st.Word("*")]))]),
            tree
        )


class TestIfStatement(TestCase):
    def test_just_if(self) -> None:
        tree = Parser([
            Token(TokenType.KEYWORD, "if", L),
            Token(TokenType.NUMBER, "1", L),
            Token(TokenType.KEYWORD, "then", L),
            Token(TokenType.NUMBER, "1", L),
            Token(TokenType.PUNCTUATION, ";", L)]).parse_if()

        self.assertEqual(
            st.If(st.Expression([st.IntLiteral(1)]), st.Expression([st.IntLiteral(1)]), None),
            tree
        )

    def test_if_else_chain(self) -> None:
        tree = Parser([
            Token(TokenType.KEYWORD, "if", L),
            Token(TokenType.NUMBER, "0", L),
            Token(TokenType.KEYWORD, "then", L),
            Token(TokenType.NUMBER, "0", L),
            Token(TokenType.KEYWORD, "else-if", L),
            Token(TokenType.NUMBER, "1", L),
            Token(TokenType.KEYWORD, "then", L),
            Token(TokenType.NUMBER, "1", L),
            Token(TokenType.KEYWORD, "else", L),
            Token(TokenType.NUMBER, "2", L),
            Token(TokenType.PUNCTUATION, ";", L)]).parse_if()

        self.assertEqual(
            st.If(
                cond=st.Expression([st.IntLiteral(0)]),
                body=st.Expression([st.IntLiteral(0)]),
                els=st.Expression([st.If(
                    cond=st.Expression([st.IntLiteral(1)]),
                    body=st.Expression([st.IntLiteral(1)]),
                    els=st.Expression([st.IntLiteral(2)]))])),
            tree
        )


class TestWhileStatement(TestCase):
    def test_while_statement(self) -> None:
        tree = Parser([
            Token(TokenType.KEYWORD, "while", L),
            Token(TokenType.NUMBER, "1", L),
            Token(TokenType.KEYWORD, "do", L),
            Token(TokenType.NUMBER, "1", L),
            Token(TokenType.PUNCTUATION, ";", L)]).parse_while()

        self.assertEqual(
            st.While(
                cond=st.Expression([st.IntLiteral(1)]),
                body=st.Expression([st.IntLiteral(1)])),
            tree
        )


class TestLocalsStatement(TestCase):
    def test_single_local(self) -> None:
        tree = Parser([
            Token(TokenType.KEYWORD, "->", L),
            Token(TokenType.WORD, "x", L),
            Token(TokenType.KEYWORD, "do", L),
            Token(TokenType.WORD, "x", L),
            Token(TokenType.NUMBER, "1", L),
            Token(TokenType.WORD, "+", L),
            Token(TokenType.PUNCTUATION, ";", L),
            ]).parse_locals()

        self.assertEqual(
            st.Locals(names=["x"], body=st.Expression([
                st.Word("x"), st.IntLiteral(1), st.Word("+")])),
            tree
        )

    def test_many_locals(self) -> None:
        tree = Parser([
            Token(TokenType.KEYWORD, "->", L),
            Token(TokenType.WORD, "x", L),
            Token(TokenType.PUNCTUATION, ",", L),
            Token(TokenType.WORD, "y", L),
            Token(TokenType.PUNCTUATION, ",", L),
            Token(TokenType.WORD, "z", L),
            Token(TokenType.KEYWORD, "do", L),
            Token(TokenType.WORD, "x", L),
            Token(TokenType.WORD, "y", L),
            Token(TokenType.WORD, "z", L),
            Token(TokenType.WORD, "+", L),
            Token(TokenType.WORD, "+", L),
            Token(TokenType.PUNCTUATION, ";", L)]).parse_locals()

        self.assertEqual(
            st.Locals(names=["x", "y", "z"], body=st.Expression([
                st.Word("x"), st.Word("y"), st.Word("z"), st.Word("+"), st.Word("+")])),
            tree
        )


class TestParserErrors(TestCase):
    def setUp(self) -> None:
        self.row, self.col = 0, -1

    def loc(self, newline: bool = False) -> Loc:
        if newline:
            self.row += 1
            self.col = 0
        else:
            self.col += 1

        return Loc("test", self.row, self.col)

    def test_some_errors(self) -> None:
        p = Parser([
            Token(TokenType.KEYWORD, "def", self.loc()),
            Token(TokenType.WORD, "func", self.loc()),
            Token(TokenType.KEYWORD, "is", self.loc()),
            Token(TokenType.WORD, "+", self.loc(newline=True)),
            Token(TokenType.PUNCTUATION, ";", self.loc()),
            Token(TokenType.KEYWORD, "def", self.loc(newline=True)),
            Token(TokenType.EOF, "", self.loc())
        ])

        tree = p.parse()
        self.assertIsNone(tree)


if __name__ == '__main__':
    main()
