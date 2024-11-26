from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from enum import Enum
from typing import NamedTuple, Optional


class TokenType(Enum):
    """Token types recognized by the regexp-based lexer"""
    ATOM = r'[0-9A-Za-z]'
    OP = r'[-+=?*/.![\]:]'
    EOF = r'$'


class Token(NamedTuple):
    type: TokenType
    value: str


type S = Atom | Cons


class Atom(str):
    def __repr__(self):
        return self


class Cons(NamedTuple):
    char: str
    rest: tuple[S, ...]

    def __repr__(self):
        r = " ".join(str(t) for t in self.rest)
        return f"({self.char} {r})"


class Lexer:
    _scanner = re.compile("|".join(f"(?P<{t.name}>{t.value})" for t in TokenType))
    _types = tuple(TokenType)

    def __init__(self, input_: str):
        tokens = list(Lexer._scan(input_))
        tokens.reverse()
        self.tokens = tokens

    @classmethod
    def _scan(cls, input_: str) -> Iterable[Token]:
        for match in cls._scanner.finditer(input_):
            i, text = next(filter(lambda m: m[1] is not None, enumerate(match.groups())))
            yield Token(cls._types[i], text)

    def next(self) -> Token:
        return self.tokens.pop()

    def peek(self) -> Token:
        return self.tokens[-1]

    def consume(self, *args):
        token = self.next()
        assert token == args


def expr(input_: str) -> S:
    lexer = Lexer(input_)
    return expr_bp(lexer, 0)


def expr_bp(lexer: Lexer, min_bp: int) -> S:
    match lexer.next():
        case Token(TokenType.ATOM, it):
            lhs = Atom(it)
        case Token(TokenType.OP, '('):
            lhs = expr_bp(lexer, 0)
            lexer.consume(TokenType.OP, ')')
        case Token(TokenType.OP, op):
            _, r_bp = prefix_binding_power(op)
            rhs = expr_bp(lexer, r_bp)
            lhs = Cons(op, (rhs,))
        case t:
            raise RuntimeError(f"bad token: {t}")

    while True:
        match lexer.peek():
            case Token(TokenType.EOF, _):
                break
            case Token(TokenType.OP, op):
                pass
            case t:
                raise RuntimeError(f"bad token: {t}")

        if pbp := postfix_binding_power(op):
            l_bp, _ = pbp
            if l_bp < min_bp:
                break
            lexer.next()

            if op == '[':
                rhs = expr_bp(lexer, 0)
                lhs = Cons(op, (lhs, rhs))
                lexer.consume(TokenType.OP, ']')
            else:
                lhs = Cons(op, (lhs,))

        elif ibp := infix_binding_power(op):
            l_bp, r_bp = ibp
            if l_bp < min_bp:
                break
            lexer.next()

            if op == '?':
                mhs = expr_bp(lexer, 0)
                lexer.consume(TokenType.OP, ':')
                rhs = expr_bp(lexer, r_bp)
                lhs = Cons(op, (lhs, mhs, rhs))
            else:
                rhs = expr_bp(lexer, r_bp)
                lhs = Cons(op, (lhs, rhs))

        else:
            break

    return lhs


def prefix_binding_power(op: str) -> tuple[None, int]:
    bp = {'+': (None, 9), '-': (None, 9)}.get(op, None)
    if bp is not None:
        return bp
    else:
        raise RuntimeError(f"bad op: {op}")


def postfix_binding_power(op: str) -> Optional[tuple[int, None]]:
    bp = {'!': (11, None), '[': (11, None)}.get(op, None)
    return bp


def infix_binding_power(op: str) -> Optional[tuple[int, int]]:
    bp = {'=': (2, 1), '?': (4, 3), '+': (5, 6), '-': (5, 6), '*': (7, 8), '/': (7, 8), '.': (14, 13)}.get(op, None)
    return bp


def tests():
    """
    >>> expr("1")
    1
    >>> expr('+1')
    (+ 1)
    >>> expr('1 + +-1')
    (+ 1 (+ (- 1)))
    >>> expr("1 + 2 * 3")
    (+ 1 (* 2 3))
    >>> expr("a + b * c * d + e")
    (+ (+ a (* (* b c) d)) e)
    >>> expr("f . g . h")
    (. f (. g h))
    >>> expr("1 + 2 + f . g . h * 3 * 4")
    (+ (+ 1 2) (* (* (. f (. g h)) 3) 4))
    >>> expr("--1 * 2")
    (* (- (- 1)) 2)
    >>> expr("--f . g")
    (- (- (. f g)))
    >>> expr("-9!")
    (- (! 9))
    >>> expr("f . g !")
    (! (. f g))
    >>> expr("(((0)))")
    0
    >>> expr("x[0][1]")
    ([ ([ x 0) 1)
    >>> expr("a ? b :\\nc ? d\\n: e")
    (? a b (? c d e))
    >>> expr("a = 0 ? b : c = d")
    (= a (= (? 0 b c) d))
    """

    import doctest
    doctest.testmod()


def main():
    """parses input from stdin"""
    for line in sys.stdin:
        line = line.strip()
        s = expr(line)
        print(s)


if __name__ == "__main__":
    tests()
