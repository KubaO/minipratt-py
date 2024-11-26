from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, NamedTuple


class Cons(NamedTuple):
    head: str
    rest: list[Cons]

    def __repr__(self):
        if self.rest and any(self.rest):
            rest = "".join(f" {s}" for s in self.rest if s is not None)
            return f"({self.head}{rest})"
        else:
            return self.head


def lexer(input_: str) -> Iterable[Optional[str]]:
    yield from filter(lambda c: not c.isspace(), input_)
    yield None


def expr(input_: str) -> Cons:
    tokens = lexer(input_)
    return expr_bp(tokens)


@dataclass(slots=True)
class Frame:
    min_bp: int
    lhs: Optional[Cons]
    token: Optional[str]


def expr_bp(tokens: Iterable[str]) -> Cons:
    top = Frame(0, None, None)
    stack: list[Frame] = []

    for token in tokens:
        while True:
            match binding_power(token, top.lhs is None):
                case (l_bp, r_bp) if top.min_bp <= l_bp:
                    break
                case _:
                    res, top = top, stack.pop() if stack else None
                    if top is None:
                        return res.lhs

                    top.lhs = Cons(res.token, [top.lhs, res.lhs])

        if token == ')':
            assert top.token == '('
            res, top = top, stack.pop()
            top.lhs = res.lhs
            continue

        stack.append(top)
        top = Frame(min_bp=r_bp, lhs=None, token=token)


def binding_power(op: Optional[str], prefix: bool) -> Optional[tuple[int, int]]:
    match op:
        case _ if op and re.match('[0-9a-zA-Z]', op):
            return 99, 100
        case '(':
            return 99, 0
        case ')':
            return 0, 100
        case '=':
            return 2, 1
        case '+' | '-' if prefix:
            return 99, 9
        case '+' | '-':
            return 5, 6
        case '*' | '/':
            return 7, 8
        case '!':
            return 11, 100
        case '.':
            return 14, 13
        case _:
            return None


def tests():
    """
    >>> expr("1")
    1
    >>> expr("1 + 2 * 3")
    (+ 1 (* 2 3))
    >>> expr("a + b * c * d + e")
    (+ (+ a (* (* b c) d)) e)
    >>> expr("f . g . h")
    (. f (. g h))
    >>> expr(" 1 + 2 + f . g . h * 3 * 4")
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
    >>> expr("(1 + 2) * 3")
    (* (+ 1 2) 3)
    >>> expr("1 + (2 * 3)")
    (+ 1 (* 2 3))
    """

    import doctest
    doctest.testmod()


if __name__ == '__main__':
    tests()


def main():
    for line in sys.stdin:
        s = expr(line)
        print(s)
