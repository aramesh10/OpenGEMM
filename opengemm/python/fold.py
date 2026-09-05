"""Constant folding for an emitted kernel.

Once the configuration is compiled in, every `if constexpr` in the kernel
has a known answer, and so do the plain `if`s and ternaries whose conditions
are built from configuration constants. These passes take the answer: the
taken branch is spliced in where the statement stood, the other is gone,
and the helper templates the kernel instantiates with one set of arguments
become plain functions so their branches fold too.

The passes work on text, with a small string-aware scanner for braces and
statements and an expression parser that folds only what it can prove.
Anything it cannot parse is left exactly as it was.
"""

import re


PAIRS = {"(": ")", "[": "]", "{": "}"}


def skip_string(text, i):
    """Return the index after the string literal whose opening quote is at `i`.
    """
    quote = text[i]
    i += 1
    while i < len(text) and text[i] != quote:
        i += 2 if text[i] == "\\" else 1
    return i + 1


def match_bracket(text, i):
    """Return the index of the bracket matching the one at `i`."""
    stack = [PAIRS[text[i]]]
    i += 1
    while i < len(text):
        c = text[i]
        if c in "\"'":
            i = skip_string(text, i)
            continue
        if c in PAIRS:
            stack.append(PAIRS[c])
        elif c in ")]}":
            if c == stack[-1]:
                stack.pop()
            if not stack:
                return i
        i += 1
    raise ValueError("unbalanced brackets")


def skip_ws(text, i):
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def is_word(text, i, word):
    return (text.startswith(word, i)
            and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_."))
            and not (i + len(word) < len(text)
                     and (text[i + len(word)].isalnum()
                          or text[i + len(word)] == "_")))


def statement_end(text, i):
    """Return the index just past the statement starting at `i`: its `;` at
    depth zero, or the `}` of the block it ends in.
    """
    depth = 0
    while i < len(text):
        c = text[i]
        if c in "\"'":
            i = skip_string(text, i)
            continue
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "{" and depth == 0:
            j = match_bracket(text, i)
            k = skip_ws(text, j + 1)
            # A brace followed by one of these closes an initializer or a
            # lambda, not the statement.
            if k < len(text) and text[k] in ";(),.":
                i = j + 1
                continue
            return j + 1
        elif c == ";" and depth == 0:
            return i + 1
        i += 1
    return len(text)


def line_start(text, i):
    return text.rfind("\n", 0, i) + 1


def line_indent(text, i):
    start = line_start(text, i)
    return len(text[start:i]) - len(text[start:i].lstrip(" ")) \
        if text[start:i].strip() == "" else len(text[start:]) - len(text[start:].lstrip(" "))



TOKEN = re.compile(r"""\s*(?:
    (?P<num>0[xX][0-9a-fA-F]+[uUlL]*|\d+[uUlL]*)
  | (?P<id>[A-Za-z_]\w*(?:::\w+)*)
  | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<op>\|\||&&|==|!=|<=|>=|<<|>>|->|[-+*/%&|^~!<>?:(),.\[\]])
)""", re.X)

BINARY = [("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="),
          ("<", "<=", ">", ">="), ("<<", ">>"), ("+", "-"), ("*", "/", "%")]
COMPARE = {"==", "!=", "<", "<=", ">", ">="}


class Node:
    __slots__ = ("kind", "op", "kids", "span", "value", "text", "changed")

    def __init__(self, kind, op, kids, span):
        self.kind, self.op, self.kids, self.span = kind, op, kids, span
        self.value = self.text = None
        self.changed = False


class Parser:
    def __init__(self, text):
        self.src = text
        self.tokens = []
        pos = 0
        while pos < len(text):
            m = TOKEN.match(text, pos)
            if not m or m.end() == pos:
                if text[pos:].strip() == "":
                    break
                raise SyntaxError(f"bad token at {text[pos:pos + 10]!r}")
            kind = m.lastgroup
            self.tokens.append((kind, m.group(kind), m.start(kind), m.end()))
            pos = m.end()
        self.i = 0

    def peek(self, kind=None):
        if self.i >= len(self.tokens):
            return None
        tok = self.tokens[self.i]
        return tok[1] if kind is None else (tok[1] if tok[0] == kind else None)

    def take(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def expect(self, value):
        if self.peek() != value:
            raise SyntaxError(f"expected {value!r}")
        return self.take()

    def parse(self):
        node = self.ternary()
        if self.i != len(self.tokens):
            raise SyntaxError("trailing tokens")
        return node

    def ternary(self):
        cond = self.binary(0)
        if self.peek() != "?":
            return cond
        self.take()
        a = self.ternary()
        self.expect(":")
        b = self.ternary()
        return Node("tern", "?", [cond, a, b], (cond.span[0], b.span[1]))

    def binary(self, level):
        if level == len(BINARY):
            return self.unary()
        left = self.binary(level + 1)
        while self.peek() in BINARY[level] and self.tokens[self.i][0] == "op":
            op = self.take()[1]
            right = self.binary(level + 1)
            left = Node("bin", op, [left, right], (left.span[0], right.span[1]))
        return left

    def unary(self):
        if self.peek() in ("!", "-", "+", "~") and self.tokens[self.i][0] == "op":
            tok = self.take()
            child = self.unary()
            return Node("un", tok[1], [child], (tok[2], child.span[1]))
        return self.postfix()

    def postfix(self):
        node = self.primary()
        while self.peek() in ("(", "[", ".", "->"):
            if self.peek() in ("(", "["):
                close = match_bracket(self.src, self.tokens[self.i][2])
                while self.i < len(self.tokens) and self.tokens[self.i][2] <= close:
                    self.i += 1
                node = Node("opaque", None, [], (node.span[0], close + 1))
            else:
                self.take()
                tok = self.take()
                node = Node("opaque", None, [], (node.span[0], tok[3]))
        return node

    def primary(self):
        tok = self.peek()
        if tok is None:
            raise SyntaxError("unexpected end")
        kind, value, start, end = self.take()
        if kind == "num":
            return Node("num", value, [], (start, end))
        if kind == "id":
            return Node("id", value, [], (start, end))
        if kind == "str":
            return Node("opaque", None, [], (start, end))
        if value == "(":
            inner = self.ternary()
            close = self.expect(")")
            return Node("paren", None, [inner], (start, close[3]))
        raise SyntaxError(f"unexpected {value!r}")


def c_div(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def c_mod(a, b):
    return a - b * c_div(a, b)


ARITH = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
         "*": lambda a, b: a * b, "/": c_div, "%": c_mod,
         "&": lambda a, b: a & b, "|": lambda a, b: a | b,
         "^": lambda a, b: a ^ b, "<<": lambda a, b: a << b,
         ">>": lambda a, b: a >> b,
         "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
         "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
         ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def simplify(node, src, known):
    """Fill `node.value` (a known constant, or None) and `node.text` (the
    expression as it should now read).
    """
    original = src[node.span[0]:node.span[1]]
    node.text = original
    if node.kind == "num":
        digits = node.op.rstrip("uUlL")
        node.value = int(digits, 0)
    elif node.kind == "id":
        if node.op in known:
            node.value = known[node.op]
    elif node.kind == "paren":
        inner = node.kids[0]
        simplify(inner, src, known)
        node.value = inner.value
        if inner.changed:
            node.changed = True
            bare = re.fullmatch(r"[\w:]+", inner.text) or inner.text in ("true", "false")
            node.text = inner.text if bare else "(" + inner.text + ")"
    elif node.kind == "un":
        child = node.kids[0]
        simplify(child, src, known)
        if child.value is not None:
            if node.op == "!":
                node.value = not child.value
                node.text = literal(node.value)
                node.changed = True
            elif node.op == "-":
                node.value = -child.value
            elif node.op == "+":
                node.value = child.value
            elif node.op == "~":
                node.value = ~child.value
        elif child.changed:
            node.changed = True
            node.text = node.op + child.text
    elif node.kind == "bin":
        left, right = node.kids
        simplify(left, src, known)
        simplify(right, src, known)
        if node.op == "&&" or node.op == "||":
            short, other = (False, True) if node.op == "&&" else (True, False)
            for a, b in ((left, right), (right, left)):
                if a.value is not None and bool(a.value) == short:
                    node.value, node.text, node.changed = short, literal(short), True
                    break
                if a.value is not None and bool(a.value) == other:
                    node.value, node.text, node.changed = b.value, b.text, True
                    if b.value is not None:
                        node.value = bool(b.value)
                    break
            else:
                if left.changed or right.changed:
                    node.changed = True
                    node.text = f"{left.text} {node.op} {right.text}"
        elif left.value is not None and right.value is not None:
            node.value = ARITH[node.op](left.value, right.value)
            if node.op in COMPARE:
                node.text = literal(node.value)
                node.changed = True
            elif left.changed or right.changed:
                node.changed = True
                node.text = f"{left.text} {node.op} {right.text}"
        elif left.changed or right.changed:
            node.changed = True
            node.text = f"{left.text} {node.op} {right.text}"
    elif node.kind == "tern":
        cond, a, b = node.kids
        simplify(cond, src, known)
        simplify(a, src, known)
        simplify(b, src, known)
        if cond.value is not None:
            chosen = a if cond.value else b
            node.value, node.text, node.changed = chosen.value, chosen.text, True
        elif cond.changed or a.changed or b.changed:
            node.changed = True
            node.text = f"{cond.text} ? {a.text} : {b.text}"
    return node


def simplify_expression(expr, known):
    """Fold an expression string as far as its known names allow.

    Returns:
        `(value, text)`: the constant it evaluates to, or None, and the folded
        text. Unparseable text comes back unchanged with no value.
    """
    try:
        node = Parser(expr).parse()
    except (SyntaxError, ValueError, IndexError):
        return None, expr
    simplify(node, expr, known)
    if node.value is not None and isinstance(node.value, bool):
        return node.value, literal(node.value)
    return node.value, node.text if node.changed else expr


def truth(value):
    return None if value is None else bool(value)



class If:
    __slots__ = ("start", "cond", "body", "else_body", "end", "constexpr")


def parse_body(text, i):
    """Return `(kind, start, end, node)` for the statement at `i`: a block, a
    nested if, or a single statement.
    """
    k = skip_ws(text, i)
    if text[k] == "{":
        return ("block", k, match_bracket(text, k) + 1, None)
    if is_word(text, k, "if"):
        node = parse_if(text, k)
        return ("if", k, node.end, node)
    return ("stmt", k, statement_end(text, k), None)


def parse_if(text, i):
    node = If()
    node.start = i
    m = re.match(r"if\s*(constexpr\b)?\s*\(", text[i:])
    if not m:
        raise ValueError("not an if")
    node.constexpr = bool(m.group(1))
    p = i + m.end() - 1
    q = match_bracket(text, p)
    node.cond = (p + 1, q)
    node.body = parse_body(text, q + 1)
    node.end = node.body[2]
    node.else_body = None
    k = skip_ws(text, node.end)
    if is_word(text, k, "else"):
        node.else_body = parse_body(text, k + 4)
        node.end = node.else_body[2]
    return node


def dedent_to(block_text, indent):
    """Return a block's inner text as statements at `indent`, the first line
    bare since it lands where the `if` keyword stood.
    """
    lines = block_text.strip("\n").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    inner = min(len(l) - len(l.lstrip(" ")) for l in lines if l.strip())
    shift = max(inner - indent, 0)
    out = [l[shift:] if l.startswith(" " * shift) else l.lstrip(" ")
           for l in lines]
    out[0] = out[0].lstrip(" ")
    return "\n".join(out)


def reindent_statement(stmt, from_indent, to_indent):
    """Return a single statement with its continuation lines moved along with
    its first.
    """
    lines = stmt.split("\n")
    shift = max(from_indent - to_indent, 0)
    return "\n".join([lines[0].lstrip(" ")]
                     + [l[shift:] if l.startswith(" " * shift) else l.lstrip(" ")
                        for l in lines[1:]])


def ends_in_return(stmts):
    return re.search(r"(?:^|[;}\n])\s*return\b[^;]*;\s*$", stmts) is not None


def fold_body(text, body, known, hoist, indent):
    """Return the folded text of an if body.

    With `hoist` the body is unwrapped into statements at `indent`; without it
    the body keeps its shape, as an else's body or the body of a brace-less for
    must.
    """
    kind, s, e, node = body
    if kind == "block":
        inner = fold_region(text, s + 1, e - 1, known)
        return dedent_to(inner, indent) if hoist else "{" + inner + "}"
    if kind == "if":
        return rewrite_if(text, node, known, hoist)[0]
    stmt = fold_region(text, s, e, known)
    return reindent_statement(stmt, line_indent(text, s), indent) if hoist else stmt


def rewrite_if(text, node, known, hoist):
    """Return the replacement text for a whole if statement, and whether it
    ended in a return that makes the rest of its block unreachable.
    """
    indent = line_indent(text, node.start)
    value, cond = simplify_expression(text[node.cond[0]:node.cond[1]], known)
    value = truth(value)
    if value is True:
        out = fold_body(text, node.body, known, hoist, indent)
        return out, hoist and ends_in_return(out)
    if value is False:
        if node.else_body is None:
            return "", False
        out = fold_body(text, node.else_body, known, hoist, indent)
        return out, hoist and ends_in_return(out)
    head = text[node.start:node.cond[0]] + cond + ")"
    head += text[node.cond[1] + 1:node.body[1]]
    head += fold_body(text, node.body, known, False, indent)
    if node.else_body is not None:
        tail = fold_body(text, node.else_body, known, False, indent)
        if tail:
            head += text[node.body[2]:node.else_body[1]] + tail
    return head, False


def fold_region(text, lo, hi, known):
    """Fold every if statement in `text[lo:hi]`, recursing into blocks."""
    out = []
    pos = i = lo
    while i < hi:
        c = text[i]
        if c in "\"'":
            i = skip_string(text, i)
            continue
        if c == "{":
            j = match_bracket(text, i)
            out.append(text[pos:i + 1])
            out.append(fold_region(text, i + 1, j, known))
            pos = i = j
            continue
        if c == "i" and is_word(text, i, "if"):
            before = text[:i].rstrip()
            # Only an if at statement level can be hoisted into its block; one
            # inside an expression keeps its braces.
            statement_level = (not before or before[-1] in ";{}"
                               or before.endswith("#pragma unroll"))
            node = parse_if(text, i)
            new, terminates = rewrite_if(text, node, known, statement_level)
            start, end = i, node.end
            if new == "":
                ls = line_start(text, i)
                if text[ls:i].strip() == "":
                    start = ls
                le = text.find("\n", end)
                if le != -1 and text[end:le].strip() == "":
                    end = le + 1
            out.append(text[pos:start])
            out.append(new)
            pos = i = end
            if terminates:
                return "".join(out) + "\n"
            continue
        i += 1
    out.append(text[pos:hi])
    return "".join(out)


def fold_ifs(text, known):
    return fold_region(text, 0, len(text), known)



def expression_bounds(text, i):
    """Return `(start, end)` of the expression around `i`: back to the `=`,
    `(`, `[`, `,`, `;` or `return` that opens it, forward to the `;`, `,` or
    bracket that closes it.
    """
    depth = 0
    j = i
    while j > 0:
        c = text[j - 1]
        if c in "\"'":
            k = j - 2
            while k >= 0 and text[k] != c:
                k -= 1
            j = k
            continue
        if c in ")]}":
            depth += 1
        elif c in "([{":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and c in ";,?:":
            break
        # A lone `=` opens the expression; `==`, `!=`, `<=` and `>=` are inside
        # it.
        elif depth == 0 and c == "=" and text[j - 2] not in "=!<>" \
                and text[j] != "=":
            break
        j -= 1
    start = skip_ws(text, j)
    if text[start:].startswith("return"):
        start = skip_ws(text, start + 6)
    depth = 0
    k = i
    while k < len(text):
        c = text[k]
        if c in "\"'":
            k = skip_string(text, k)
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and c in ";,":
            break
        k += 1
    return start, k


def fold_expressions(text, known):
    """Fold the ternaries and logical operators whose operands are known,
    wherever they stand.
    """
    out, pos, i = [], 0, 0
    while i < len(text):
        c = text[i]
        if c in "\"'":
            i = skip_string(text, i)
            continue
        if c == "?" or text.startswith("&&", i) or text.startswith("||", i):
            start, end = expression_bounds(text, i)
            if start < pos or start >= end:
                i += 1
                continue
            expr = text[start:end]
            value, new = simplify_expression(expr, known)
            if new != expr and value is not None or new != expr:
                out.append(text[pos:start])
                out.append(new.rstrip())
                pos = end
            i = end
            continue
        i += 1
    out.append(text[pos:])
    return "".join(out)



TEMPLATE_FN = re.compile(
    r"^template <([^>]*)>\s*\n((?:(?:__\w+|inline|static|constexpr)\s+)+)"
    r"([\w:]+(?:\s*[*&])?)\s+(\w+)\s*\(", re.M)


def function_end(text, open_paren):
    """Return the index after the closing brace of the function whose parameter
    list opens at `open_paren`.
    """
    close = match_bracket(text, open_paren)
    brace = text.index("{", close)
    return match_bracket(text, brace) + 1


def split_args(text):
    parts, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c in "(<[":
            depth += 1
        elif c in ")>]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def call_sites(text, name, exclude):
    """Return `[(start, end_of_args, [arg texts])]` for every `NAME<...>(`
    outside the definition span `exclude`.
    """
    sites = []
    for m in re.finditer(rf"\b{name}\s*<", text):
        if exclude[0] <= m.start() < exclude[1]:
            continue
        open_angle = m.end() - 1
        close = text.find(">", open_angle)
        after = skip_ws(text, close + 1)
        if close < 0 or after >= len(text) or text[after] != "(":
            continue
        sites.append((m.start(), close + 1, split_args(text[open_angle + 1:close])))
    return sites


def monomorphize(text, known):
    """Bind the template parameters of helpers every caller agrees on, and
    clone the helpers callers disagree on when a branch depends on them.
    """
    changed = True
    while changed:
        changed = False
        for m in TEMPLATE_FN.finditer(text):
            params = [p.split()[-1] for p in split_args(m.group(1))]
            types = [p.split()[0] for p in split_args(m.group(1))]
            value_params = [p for p, t in zip(params, types)
                            if t not in ("class", "typename")]
            if not value_params:
                continue
            name = m.group(4)
            start, end = m.start(), function_end(text, m.end() - 1)
            sites = call_sites(text, name, (start, end))
            if not sites:
                continue
            bound = {}
            for _, _, args in sites:
                if len(args) < len(value_params):
                    break
                values = []
                for arg in args[:len(value_params)]:
                    value, _ = simplify_expression(arg, known)
                    if value is None:
                        break
                    values.append(value)
                else:
                    bound.setdefault(tuple(values), []).append(args)
                    continue
                break
            else:
                body = text[start:end]
                # Callers that disagree only need clones when a branch depends
                # on the parameter.
                if len(bound) > 1 and "if constexpr" not in body:
                    continue
                clones = []
                for values, spellings in bound.items():
                    suffix = "" if len(bound) == 1 else "_" + "_".join(
                        re.sub(r"\W", "", literal(v)) for v in values)
                    clone = body
                    for index, (param, value) in enumerate(zip(value_params, values)):
                        spelled = next((s[index] for s in spellings
                                        if re.fullmatch(r"\w+", s[index])),
                                       literal(value))
                        clone = re.sub(rf"\b{param}\b", spelled, clone)
                    kept = [p for p in split_args(m.group(1))
                            if p.split()[-1] not in value_params]
                    header = f"template <{', '.join(kept)}>\n" if kept else ""
                    clone = header + clone[clone.index("\n") + 1:]
                    if suffix:
                        clone = re.sub(rf"\b{name}\s*\(", f"{name}{suffix}(",
                                       clone, count=1)
                    clones.append((values, suffix, clone))
                rewritten = []
                pos = 0
                for site_start, site_end, args in sites:
                    values = tuple(simplify_expression(a, known)[0]
                                   for a in args[:len(value_params)])
                    suffix = next(s for v, s, _ in clones if v == values)
                    rest = args[len(value_params):]
                    rewritten.append(text[pos:site_start])
                    rewritten.append(f"{name}{suffix}"
                                     + (f"<{', '.join(rest)}>" if rest else ""))
                    pos = site_end
                rewritten.append(text[pos:])
                text = "".join(rewritten)
                shift = len(text) - (len(rewritten[-1]) + sum(
                    len(r) for r in rewritten[:-1]))
                new_start = text.index(body) if body in text else None
                if new_start is None:
                    raise RuntimeError(f"lost the definition of {name}")
                text = (text[:new_start] + "\n\n".join(c for _, _, c in clones)
                        + text[new_start + len(body):])
                changed = True
                break
    return text



FN_DEF = re.compile(
    r"^(?:template <[^>]*>\s*\n)?(?:(?:__device__|__host__|__forceinline__|"
    r"inline|static|constexpr)\s+)+[\w:]+(?:\s*[*&])?\s+(\w+)\s*\(", re.M)


def prune_functions(header, source, keep):
    """Drop every function definition in the header that nothing references in
    either file, to a fixpoint.
    """
    while True:
        dropped = False
        for m in FN_DEF.finditer(header):
            name = m.group(1)
            if name in keep:
                continue
            uses = len(re.findall(rf"\b{name}\b", header + source))
            if uses > 1:
                continue
            end = function_end(header, m.end() - 1)
            while end < len(header) and header[end] == "\n":
                end += 1
            header = header[:m.start()] + header[end:]
            dropped = True
            break
        if not dropped:
            return header


CONST_DECL = re.compile(
    r"^[ \t]*(?:const|constexpr) [\w:]+ (\w+)\s*=([^;]*);[ \t]*\n", re.M)


def prune_constants(header, source, keep=()):
    """Drop the const and constexpr declarations nothing reads, file-scope and
    local alike, to a fixpoint; a local whose initializer calls something is
    left alone.
    """
    while True:
        dropped = False
        for m in CONST_DECL.finditer(header):
            name, init = m.group(1), m.group(2)
            if name in keep or re.search(r"\b\w+\s*\(", init):
                continue
            if len(re.findall(rf"\b{name}\b", header + source)) == 1:
                header = header[:m.start()] + header[m.end():]
                dropped = True
                break
        if not dropped:
            return header


def tidy(text):
    """Clean up after folding: a literal or name left in parentheses by a
    folded ternary, and the blank lines a removed branch leaves behind.
    """
    text = re.sub(r"(?<=[\[(=+\-*/%&|^?:,])(\s*)\((true|false|-?\d+|[A-Za-z_]\w*)\)"
                  r"(?=\s*(?:[;\]\),+\-*/%&|^?:]|$))", r"\1\2", text, flags=re.M)
    text = re.sub(r"\n[ \t]*\n([ \t]*\n)+", "\n\n", text)
    text = re.sub(r"\{\n[ \t]*\n", "{\n", text)
    text = re.sub(r"\n[ \t]*\n([ \t]*\})", r"\n\1", text)
    return text


def known_constants(header, extra=None):
    """Return every file-scope `constexpr <type> NAME = <literal>;` as a value,
    on top of `extra`.
    """
    known = dict(extra or {})
    for kind, name, value in re.findall(
            r"^constexpr ([\w:]+) (\w+)\s*=\s*([\w:]+)\s*;", header, re.M):
        if value in ("true", "false"):
            known[name] = value == "true"
        elif re.fullmatch(r"-?\d+[uUlL]*|0[xX][0-9a-fA-F]+", value):
            known[name] = int(value.rstrip("uUlL"), 0)
        elif value in known:
            known[name] = known[value]
    return known


def fold_all(header, source, known, keep_functions):
    """Fold, monomorphize and prune the pair to a fixpoint.

    Args:
        header: Header text.
        source: Source text.
        known: Constants known before the header is read.
        keep_functions: Names never pruned, such as the kernel and the entry
            point.

    Returns:
        `(header, source)`, folded.
    """
    # Far more rounds than a kernel has needed; the loop exits at the first
    # quiet one.
    for _ in range(12):
        before = (header, source)
        known = known_constants(header, known)
        header = fold_expressions(fold_ifs(header, known), known)
        source = fold_expressions(fold_ifs(source, known), known)
        both = monomorphize(header + "\n\x00\n" + source, known)
        header, source = both.split("\n\x00\n")
        header = prune_functions(header, source, keep_functions)
        header = prune_constants(header, source)
        if (header, source) == before:
            break
    return tidy(header), tidy(source)
