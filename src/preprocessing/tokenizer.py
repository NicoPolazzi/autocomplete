import ast
import re
import warnings

from src.logger import get_logger

logger = get_logger(__name__)


def post_tokenization(codes: list[str]):

    for code_snippet in codes:
        clean_code = remove_triple_backticks_and_comments(code_snippet)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            result = _tokenize(clean_code)
            yield from result


def _tokenize(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    visitor = _TokenVisitor()
    visitor.visit(tree)

    return visitor.tokens


def remove_triple_backticks_and_comments(code: str) -> str:
    code = re.sub(r"^```python\n", "", code)
    code = re.sub(r"```$", "", code)
    code = re.sub(r"#.*\n", "", code)
    return code


class _TokenVisitor(ast.NodeVisitor):
    tokens: list[str] = []

    def visit_FunctionDef(self, node):
        self.tokens.append("def")
        self.tokens.append(node.name)
        self.tokens.append("(")
        self.visit(node.args)
        self.tokens.append(")")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.tokens.append("class")
        self.tokens.append(node.name)
        self.tokens.append(":")
        self.generic_visit(node)

    def visit_arguments(self, node):
        for arg in node.args:
            self.tokens.append(arg.arg)
            self.tokens.append(",")
        if self.tokens and self.tokens[-1] == ",":
            self.tokens.pop()  # Remove the last comma

    def visit_Name(self, node):
        self.tokens.append(node.id)

    def visit_Constant(self, node):
        self.tokens.append(repr(node.value))

    def visit_Call(self, node):
        self.visit(node.func)
        self.tokens.append("(")
        for arg in node.args:
            self.visit(arg)
            self.tokens.append(",")
        if self.tokens and self.tokens[-1] == ",":
            self.tokens.pop()  # Remove the last comma
        self.tokens.append(")")

    def visit_Attribute(self, node):
        self.visit(node.value)
        self.tokens.append(".")
        self.tokens.append(node.attr)

    def visit_Assign(self, node):
        for target in node.targets:
            self.visit(target)
        self.tokens.append("=")
        self.visit(node.value)

    def visit_Expr(self, node):
        self.visit(node.value)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.tokens.append(self.get_op_symbol(node.op))
        self.visit(node.right)

    def get_op_symbol(self, op):
        if isinstance(op, ast.Add):
            return "+"
        elif isinstance(op, ast.Sub):
            return "-"
        elif isinstance(op, ast.Mult):
            return "*"
        elif isinstance(op, ast.Div):
            return "/"
        elif isinstance(op, ast.Mod):
            return "%"
        elif isinstance(op, ast.Pow):
            return "**"
        elif isinstance(op, ast.LShift):
            return "<<"
        elif isinstance(op, ast.RShift):
            return ">>"
        elif isinstance(op, ast.BitOr):
            return "|"
        elif isinstance(op, ast.BitXor):
            return "^"
        elif isinstance(op, ast.BitAnd):
            return "&"
        elif isinstance(op, ast.FloorDiv):
            return "//"
        else:
            return ""
