from __future__ import annotations

from dataclasses import dataclass

from src.lexer import Loc


@dataclass
class Node:
    location: Loc


@dataclass
class Program:
    definitions: list[Definition]


@dataclass
class Definition(Node):
    ...


@dataclass
class WordDef(Definition):
    name: str
    body: Expression


@dataclass
class Expression:
    elements: list[Literal | Word | Statement]


@dataclass
class Literal(Node):
    ...


@dataclass
class Int(Literal):
    value: int


@dataclass
class Char(Literal):
    value: str


@dataclass
class String(Literal):
    value: str


@dataclass
class Word(Node):
    name: str


@dataclass
class Statement(Node):
    ...


@dataclass
class If(Statement):
    cond: Expression
    body: Expression
    els: Expression | None


@dataclass
class While(Statement):
    cond: Expression
    body: Expression


@dataclass
class Locals(Statement):
    names: list[str]
    body: Expression
