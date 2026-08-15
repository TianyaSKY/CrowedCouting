"""纯标准库工具：Matlab v5 .mat 解析 + JPEG 尺寸读取。

本项目的所有 .mat 真值（ShanghaiTech / UCF-QNRF / UCF-CC-50 / JHU-Crowd++）
均为 Matlab v5 格式，此模块无需 scipy/h5py 即可读取点标注。

MAT v5 格式要点（little-endian，'IM'）:
    文件头 128 字节（116 文本 + 8 子系统 + 2 版本 + 2 字节序标记）
    数据元素: [type u32][nbytes u32][data]（nbytes<=4 时内联在 8 字节 tag 内，
            否则 data 按 8 字节对齐）
    miMATRIX(14): flags / dims / name / 数据部分（PR、cell 子元素、struct 字段）
    miCOMPRESSED(15): zlib 压缩的数据元素
"""

import struct
import zlib


# ---- 数据元素类型 ----
MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13
MI_MATRIX = 14
MI_COMPRESSED = 15
MI_UTF8 = 16
MI_UTF16 = 17
MI_UTF32 = 18

# ---- 矩阵类别（flags & 0xFF，MAT 文件格式定义） ----
MX_CELL = 1
MX_STRUCT = 2
MX_OBJECT = 3
MX_CHAR = 4
MX_SPARSE = 5
MX_DOUBLE = 6
MX_SINGLE = 7
MX_INT8 = 8
MX_UINT8 = 9
MX_INT16 = 10
MX_UINT16 = 11
MX_INT32 = 12
MX_UINT32 = 13
MX_INT64 = 14
MX_UINT64 = 15


class MatV5Error(ValueError):
    """.mat 解析失败。"""


class _Reader:
    def __init__(self, data: bytes, endian: str = None, header: bool = True):
        self.data = data
        self.pos = 128 if header else 0
        if endian is None:
            if data[126:128] == b"IM":
                self.endian = "<"
            elif data[126:128] == b"MI":
                self.endian = ">"
            else:
                raise MatV5Error("未知的 .mat 字节序标记")
        else:
            self.endian = endian

    def read_tag(self):
        """返回 (type, nbytes, data, next_pos)；不推进 self.pos。

        两种元素格式：
        - 常规: [type u32][byte_count u32][data 对齐到 8 字节]
        - SDE 小数据元素: 前 4 字节高 16 位 = byte_count(1..4)、低 16 位 =
          type，数据在后 4 字节中（scipy 实测规则，与官方文档略有出入）。
        """
        if self.pos + 8 > len(self.data):
            raise MatV5Error("数据元素越界")
        first, second = struct.unpack(
            self.endian + "II", self.data[self.pos : self.pos + 8]
        )
        if first >> 16:
            # SDE
            mdtype = first & 0xFFFF
            byte_count = first >> 16
            payload = self.data[self.pos + 4 : self.pos + 4 + byte_count]
            next_pos = self.pos + 8
        else:
            mdtype = first
            byte_count = second
            start = self.pos + 8
            padded = (byte_count + 7) // 8 * 8
            payload = self.data[start : start + byte_count]
            next_pos = self.pos + 8 + padded  # byte_count=0 时只占 8 字节
        return mdtype, byte_count, payload, next_pos

    def read_tag_advance(self):
        """读 tag 并推进 self.pos，返回 (type, nbytes, data)。"""
        etype, nbytes, payload, next_pos = self.read_tag()
        self.pos = next_pos
        return etype, nbytes, payload

    def read_element(self):
        """读一个数据元素，返回解析后的 Python 对象。"""
        etype, nbytes, payload, next_pos = self.read_tag()
        self.pos = next_pos
        if etype == MI_COMPRESSED:
            raw = zlib.decompress(payload)
            sub = _Reader(raw, endian=self.endian, header=False)
            return sub.read_element()
        if etype == MI_MATRIX:
            return self._parse_matrix(payload)
        return self._parse_plain(etype, nbytes, payload)

    def _parse_plain(self, etype, nbytes, payload):
        fmt = {
            MI_INT8: "b", MI_UINT8: "B", MI_INT16: "h", MI_UINT16: "H",
            MI_INT32: "i", MI_UINT32: "I", MI_INT64: "q", MI_UINT64: "Q",
            MI_SINGLE: "f", MI_DOUBLE: "d",
        }
        if etype == MI_UTF8 or etype == MI_INT8:
            return payload.decode("utf-8", "ignore").rstrip("\x00")
        if etype not in fmt:
            return payload  # 未知类型原样返回字节
        count = nbytes // struct.calcsize(fmt[etype])
        return list(
            struct.unpack(self.endian + fmt[etype] * count, payload[:nbytes])
        )

    def _parse_matrix(self, payload):
        sub = _Reader(payload, endian=self.endian, header=False)

        # array flags
        ftype, fnbytes, fpayload = sub.read_tag_advance()
        if ftype != MI_UINT32 or fnbytes < 8:
            raise MatV5Error("array flags 格式异常")
        flags = struct.unpack(self.endian + "II", fpayload[:8])
        cls = flags[0] & 0xFF
        is_sparse = bool(flags[0] & 0x0400)
        is_complex = bool(flags[0] & 0x0800)

        # dimensions
        dtype, dnbytes, dpayload = sub.read_tag_advance()
        dims = list(
            struct.unpack(
                self.endian + "i" * (dnbytes // 4), dpayload[:dnbytes]
            )
        )
        if not dims:
            dims = [1]

        # name
        ntype, nnbytes, npayload = sub.read_tag_advance()
        name = (
            npayload[:nnbytes].decode("utf-8", "ignore").rstrip("\x00")
            if nnbytes
            else ""
        )

        count = 1
        for d in dims:
            count *= d

        # 数据部分
        if cls == MX_CELL:
            return {
                "__class__": "cell",
                "__name__": name,
                "__dims__": dims,
                "values": [sub.read_element() for _ in range(count)],
            }
        if cls == MX_STRUCT:
            if is_sparse:
                raise MatV5Error("暂不支持 sparse struct")
            # 字段名宽度 + 字段名字符串（nfields = 总长度 / 宽度）
            lt, ln, lpayload = sub.read_tag_advance()
            namelength = struct.unpack(
                self.endian + "i", lpayload[:4]
            )[0] if ln >= 4 else 0
            ft, fn, fpayload = sub.read_tag_advance()
            field_names = []
            if namelength > 0:
                for i in range(fn // namelength):
                    chunk = fpayload[
                        i * namelength : (i + 1) * namelength
                    ]
                    field_names.append(
                        chunk.decode("utf-8", "ignore").rstrip("\x00")
                    )
            # 结构体数组按列优先存储：元素 0 的所有字段，元素 1 的所有字段……
            elements = []
            for _ in range(count):
                fields = {}
                for fname in field_names:
                    fields[fname] = sub.read_element()
                elements.append(fields)
            return {
                "__class__": "struct",
                "__name__": name,
                "__dims__": dims,
                "values": elements,
            }
        if cls == MX_DOUBLE or cls == MX_SINGLE or (
            MX_INT8 <= cls <= MX_UINT64
        ):
            # 实部 PR：按 PR 元素自身的类型解包（MATLAB 可能用 uint16 存标量）
            ptype, pnbytes, ppayload = sub.read_tag_advance()
            if is_complex:
                # 读掉虚部
                sub.read_tag_advance()
            fmt = {
                MI_DOUBLE: "d", MI_SINGLE: "f", MI_INT8: "b", MI_UINT8: "B",
                MI_INT16: "h", MI_UINT16: "H", MI_INT32: "i", MI_UINT32: "I",
                MI_INT64: "q", MI_UINT64: "Q",
            }.get(ptype)
            if fmt is None:
                raise MatV5Error(f"PR 元素类型不支持: {ptype}")
            values = list(
                struct.unpack(
                    self.endian + fmt * count, ppayload[: pnbytes]
                )
            )
            if len(dims) == 2:
                rows, cols = dims
                # MATLAB 列优先 -> 转成行优先的二维列表
                matrix = [
                    [values[r * cols + c] for c in range(cols)]
                    for r in range(rows)
                ]
            else:
                matrix = values
            return {
                "__class__": "matrix",
                "__name__": name,
                "__dims__": dims,
                "values": matrix,
            }
        raise MatV5Error(f"不支持的矩阵类别: {cls}")


def loadmat_v5(path):
    """读取 Matlab v5 .mat 文件，返回 {变量名: 结构}。

    结构统一为 dict:
        matrix: {'__class__': 'matrix', '__name__', '__dims__', 'values'}
                values 为二维列表（行优先）或一维列表
        cell:   {'__class__': 'cell', 'values': [...]}
        struct: {'__class__': 'struct', 'values': [字段dict, ...]}
    顶层变量本身也是 miMATRIX，返回值去掉包装。
    """
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 128 or not data.startswith(b"MATLAB"):
        raise MatV5Error(f"不是有效的 Matlab v5 文件: {path}")
    reader = _Reader(data)
    result = {}
    while reader.pos + 8 <= len(data):
        etype, nbytes, payload, next_pos = reader.read_tag()
        reader.pos = next_pos
        if etype == MI_COMPRESSED:
            raw = zlib.decompress(payload)
            sub = _Reader(raw, endian=reader.endian, header=False)
            obj = sub.read_element()
        elif etype == MI_MATRIX:
            obj = reader._parse_matrix(payload)
        else:
            obj = reader._parse_plain(etype, nbytes, payload)
            if etype not in (MI_UTF8, MI_INT8) or not isinstance(obj, str):
                continue
        if isinstance(obj, dict) and obj.get("__class__") in (
            "matrix",
            "cell",
            "struct",
        ):
            name = obj.get("__name__") or f"var_{len(result)}"
            result[name] = obj
    return result


def matrix_points(obj):
    """从 matrix 结构取 [[x, y], ...] 点列表（兼容 1 维/2 维）。"""
    values = obj["values"]
    dims = obj["__dims__"]
    if len(dims) == 2 and dims[1] == 2:
        return [[float(r[0]), float(r[1])] for r in values]
    # Nx1 或 1xN
    flat = [float(v) for v in values]
    return [[flat[i], flat[i + 1]] for i in range(0, len(flat) - 1, 2)]


def mat_points(path):
    """通用 .mat 点标注读取，返回 [[x, y], ...]（无标注时返回空列表）。

    自动识别常见布局：
    - 顶层 Nx2 矩阵: annPoints / point / points / loc
    - struct/cell 包装: image_info.location / .point / .points（ShanghaiTech）
    - 其他任意 Nx2 顶层矩阵
    """
    m = loadmat_v5(path)
    for name in ("annPoints", "point", "points", "loc"):
        v = m.get(name)
        if v is not None and v["__class__"] == "matrix":
            pts = matrix_points(v)
            if pts:
                return pts
    info = m.get("image_info")
    if info is not None:
        if info["__class__"] == "cell":
            structs = [
                v for v in info["values"]
                if isinstance(v, dict) and v.get("__class__") == "struct"
            ]
            if structs:
                info = structs[0]
        if info["__class__"] == "struct":
            for fname in ("location", "point", "points", "loc"):
                for element in info["values"]:
                    value = element.get(fname)
                    if value is None:
                        continue
                    if value["__class__"] == "matrix":
                        return matrix_points(value)
                    if value["__class__"] == "cell":
                        pts = []
                        for sub in value["values"]:
                            if sub["__class__"] == "matrix":
                                pts.extend(matrix_points(sub))
                        return pts
    # 兜底：任意 Nx2 顶层矩阵
    for name, v in m.items():
        if (
            v["__class__"] == "matrix"
            and len(v["__dims__"]) == 2
            and v["__dims__"][1] == 2
        ):
            return matrix_points(v)
    return []


def struct_field_points(struct_obj, field):
    """从 struct（或包裹 struct 的 cell）数组取某字段的点列表。

    字段值可以是 cell（内含 matrix）或直接 matrix。
    """
    if struct_obj["__class__"] == "cell":
        # 顶层变量可能是 1x1 cell 包裹 struct（ShanghaiTech 部分版本）
        candidates = [
            v for v in struct_obj["values"]
            if isinstance(v, dict) and v.get("__class__") == "struct"
        ]
        if not candidates:
            return []
        struct_obj = candidates[0]
    points = []
    for element in struct_obj["values"]:
        value = element[field]
        if value["__class__"] == "cell":
            for sub in value["values"]:
                if sub["__class__"] == "matrix":
                    points.extend(matrix_points(sub))
        elif value["__class__"] == "matrix":
            points.extend(matrix_points(value))
    return points


def jpeg_size(path):
    """读取图片宽高（JPEG SOF / PNG IHDR，无需 PIL/cv2）。

    注: JHU-Crowd++ 官方数据中有个别实际为 PNG 但扩展名为 .jpg 的文件。
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        # PNG: 8 字节签名 + 4 长度 + IHDR，宽高在偏移 16/20（big-endian）
        if len(data) < 24 or data[12:16] != b"IHDR":
            raise ValueError(f"PNG 头部异常: {path}")
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return width, height
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"不是 JPEG/PNG 文件: {path}")
    sof = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    i = 2
    while i + 4 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        # 跳过填充字节（0xFF 0xFF），定位真正的标记码
        j = i
        while j < len(data) and data[j] == 0xFF:
            j += 1
        if j + 1 >= len(data):
            break
        marker = data[j]
        if marker == 0xD9 or marker == 0xDA:  # EOI / SOS（SOF 必在其前）
            break
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            i = j + 1
            continue
        length = int.from_bytes(data[j + 1 : j + 3], "big")
        if marker in sof:
            height = int.from_bytes(data[j + 4 : j + 6], "big")
            width = int.from_bytes(data[j + 6 : j + 8], "big")
            return width, height
        i = j + 1 + length  # 长度字段含自身 2 字节
    raise ValueError(f"JPEG 中未找到 SOF 段: {path}")
