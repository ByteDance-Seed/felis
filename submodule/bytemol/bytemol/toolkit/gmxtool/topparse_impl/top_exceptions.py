# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


class TopoDuplicateNameAtomTypeException(Exception):

    def __init__(self, name: str) -> None:
        msg = f"Duplicate name atomtype {name}"
        super().__init__(msg)


class TopoDuplicateNameMoleculeTypeException(Exception):

    def __init__(self, name: str) -> None:
        msg = f"Duplicate name moleculetype {name}"
        super().__init__(msg)
