"""Conserta modelos TensorFlow (.pb) para o importador do OpenCV 4.x.

O importador de TensorFlow do OpenCV 4.14 não repassa o tamanho de bloco da
camada ``DepthToSpace`` (a que "desdobra" canais em pixels maiores, usada
no fim das redes de super-resolução) e a rede quebra com::

    DepthSpaceLayer: blocksize is required

O arquivo .pb é um ``GraphDef`` em protobuf. Aqui ele é lido em memória e
cada nó ``DepthToSpace`` ganha um atributo ``blocksize`` com o mesmo valor
do ``block_size`` original. O arquivo em disco não é alterado, e não é
preciso instalar o TensorFlow: o formato protobuf é simples de percorrer.

Referência do formato: https://protobuf.dev/programming-guides/encoding/
"""

from __future__ import annotations

from collections.abc import Iterator

# Números dos campos no .proto do TensorFlow (graph.proto e node_def.proto).
_GRAPH_NODE = 1        # GraphDef.node (repetido)
_NODE_OP = 2           # NodeDef.op
_NODE_ATTR = 5         # NodeDef.attr: map<string, AttrValue>, cada entrada é {1: chave, 2: valor}
_LENGTH_DELIMITED = 2  # tipo de campo com tamanho na frente (texto, bytes, mensagens)


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        shift += 7
        if b < 0x80:
            return value, i


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _fields(buf: bytes) -> Iterator[tuple[int, int, bytes, bytes]]:
    """Percorre uma mensagem: (campo, tipo, bytes do campo inteiro, conteúdo)."""
    i = 0
    while i < len(buf):
        start = i
        tag, i = _read_varint(buf, i)
        number, wire_type = tag >> 3, tag & 7
        if wire_type == 0:
            _, i = _read_varint(buf, i)
            value = b""
        elif wire_type == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire_type == _LENGTH_DELIMITED:
            n, i = _read_varint(buf, i)
            value, i = buf[i:i + n], i + n
        elif wire_type == 5:
            value, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"protobuf inválido (tipo de campo {wire_type})")
        yield number, wire_type, buf[start:i], value


def _message_field(number: int, payload: bytes) -> bytes:
    return _varint(number << 3 | _LENGTH_DELIMITED) + _varint(len(payload)) + payload


def fix_depth_to_space(graph: bytes) -> tuple[bytes, int]:
    """Devolve o .pb consertado e quantos nós ``DepthToSpace`` foram ajustados."""
    out = bytearray()
    fixed = 0
    for number, wire_type, raw, node in _fields(graph):
        if number != _GRAPH_NODE or wire_type != _LENGTH_DELIMITED:
            out += raw
            continue
        op, attrs = b"", {}
        for n2, _, _, value in _fields(node):
            if n2 == _NODE_OP:
                op = bytes(value)
            elif n2 == _NODE_ATTR:
                entry = {n3: bytes(v3) for n3, _, _, v3 in _fields(value)}
                attrs[entry.get(1, b"")] = entry.get(2, b"")
        if op == b"DepthToSpace" and b"block_size" in attrs and b"blocksize" not in attrs:
            entry = _message_field(1, b"blocksize") + _message_field(2, attrs[b"block_size"])
            node = bytes(node) + _message_field(_NODE_ATTR, entry)
            fixed += 1
        out += _message_field(_GRAPH_NODE, bytes(node))
    return bytes(out), fixed
