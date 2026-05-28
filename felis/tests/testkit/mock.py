# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

mock_get_visible_cuda_devices = lambda np: lambda: [str(_) for _ in range(np)]

mock_extract_state_from_netdcd = lambda *args, **kwargs: ([0] * 2000, [0] * 2000)
