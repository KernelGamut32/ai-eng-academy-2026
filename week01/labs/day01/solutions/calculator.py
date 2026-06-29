"""A simple printing-calculator class used by the OOP notebook (Lab solution)."""


class Calc:
    """A running-total calculator that records each step.

    One-argument calls operate on the running total; two-argument calls
    operate on the two operands and replace the running total. Every step is
    recorded and can be retrieved with showcalc().
    """

    def __init__(self):
        self.total = 0
        self.history = []

    def _step(self, a, op, b, result):
        self.history.append(f'{a} {op} {b} = {result}')
        self.total = result
        return result

    def add(self, a, b=None):
        if b is None:
            return self._step(self.total, '+', a, self.total + a)
        return self._step(a, '+', b, a + b)

    def sub(self, a, b=None):
        if b is None:
            return self._step(self.total, '-', a, self.total - a)
        return self._step(a, '-', b, a - b)

    def mult(self, a, b=None):
        if b is None:
            return self._step(self.total, '*', a, self.total * a)
        return self._step(a, '*', b, a * b)

    def div(self, a, b=None):
        if b is None:
            return self._step(self.total, '/', a, self.total / a)
        return self._step(a, '/', b, a / b)

    def pow(self, a, b=None):
        if b is None:
            return self._step(self.total, '**', a, self.total ** a)
        return self._step(a, '**', b, a ** b)

    def log(self, a):
        from math import log as _log
        result = _log(a)
        self.history.append(f'log({a}) = {result}')
        self.total = result
        return result

    def showcalc(self):
        return '\n'.join(self.history)

    def __str__(self):
        return self.showcalc()

    def ac(self):
        """All clear: reset running total and history."""
        self.total = 0
        self.history = []
