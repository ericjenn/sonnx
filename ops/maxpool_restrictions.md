| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `ceil_mode` is restricted to `0` | Transient |
| `[R4]` | The padding of a spatial axis is at most the size of the kernel along that axis, i.e. $p^b_k \le k_k$ and $p^e_k \le k_k$ for every spatial axis $k$ | Transient |
| `[R5]` | The output `Indices` is not supported | Transient |

