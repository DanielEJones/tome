from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Program:
    definitions: list[Definition]


class Definition:
    ...


@dataclass
class WordDef(Definition):
    name: str
    body: Expression


@dataclass
class Expression:
    body: list[Literal | Word | Statement]


class Literal:
    ...


@dataclass
class IntLiteral(Literal):
    value: int


@dataclass
class Word:
    name: str


class Statement:
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
