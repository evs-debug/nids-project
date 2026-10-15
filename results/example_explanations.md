# Example alert explanations

For each class, the first flow in the explained sample that the model classified correctly. SHAP values are in log-odds toward the predicted class; positive pushes toward it.

## normal flow (model confidence 1.000)

| feature | value | SHAP |
|---|---|---|
| dst_host_srv_count | 255 | +1.836 |
| src_bytes | 230 | +1.365 |
| service | http | +1.112 |
| dst_bytes | 807 | +0.691 |
| dst_host_same_src_port_rate | 0 | +0.431 |

## DoS flow (model confidence 1.000)

| feature | value | SHAP |
|---|---|---|
| flag | S0 | +2.902 |
| count | 249 | +1.334 |
| dst_host_serror_rate | 1 | +1.293 |
| srv_count | 9 | +0.506 |
| same_srv_rate | 0.04 | +0.495 |

## Probe flow (model confidence 1.000)

| feature | value | SHAP |
|---|---|---|
| service | eco_i | +2.418 |
| src_bytes | 8 | +2.149 |
| protocol_type | icmp | +2.054 |
| dst_host_same_src_port_rate | 1 | +1.159 |
| dst_host_serror_rate | 0 | +0.463 |

## R2L flow (model confidence 0.939)

| feature | value | SHAP |
|---|---|---|
| hot | 2 | +2.552 |
| num_access_files | 1 | +1.322 |
| src_bytes | 51 | +1.072 |
| count | 1 | +0.583 |
| dst_host_srv_count | 255 | -0.580 |

## U2R flow (model confidence 1.000)

| feature | value | SHAP |
|---|---|---|
| root_shell | 1 | +2.758 |
| duration | 53 | +1.640 |
| dst_bytes | 3.86e+03 | +1.246 |
| service | telnet | +0.998 |
| dst_host_count | 2 | +0.773 |
