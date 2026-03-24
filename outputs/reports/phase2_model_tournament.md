# Phase 2 Model Tournament

## Serious candidates
- Temperature: `temp_et_meta, temp_rf_meta, temp_adaboost_meta, temp_mlp_meta, temp_et_meta_summary, temp_et_meta_svd`
- Abundance: `abun_sep_et_spec_noise, abun_sep_et_spec_noise_meta, abun_sep_et_spec_noise_meta_deriv, abun_shared_et_spec_noise_meta, abun_shared_mlp_svd_spec_noise_meta, alltarget_et_spec_noise_meta`
- Neural constraint: the environment has no `torch`/`tensorflow`, so the neural branch here is a compact multi-output `MLPRegressor` on SVD-compressed spectral features plus metadata.

## Temperature tournament
```
           candidate             scheme  primary_metric  mean_crps  mean_mae  mean_rmse  mean_r2  mean_nll_norm
        temp_rf_meta edge_holdout_15pct          0.9671     0.0252   18.3866    56.1532   0.9963        12.4259
        temp_et_meta edge_holdout_15pct          0.9456     0.0416   30.6023    76.8595   0.9932         2.6266
       temp_mlp_meta edge_holdout_15pct          0.9099     0.0690   53.9764   113.9119   0.9850         1.8880
  temp_adaboost_meta edge_holdout_15pct          0.9081     0.0704   64.7821    91.0265   0.9904        -0.5513
    temp_et_meta_svd edge_holdout_15pct          0.9055     0.0724   58.6658   127.8558   0.9811         1.4664
temp_et_meta_summary edge_holdout_15pct          0.8995     0.0769   62.8249   131.6780   0.9799         1.0902
        temp_et_meta       random_5fold          0.9995     0.0002    0.0016     0.3055   1.0000        -5.8573
        temp_rf_meta       random_5fold          0.9995     0.0003    0.0069     0.4980   1.0000        -5.7041
temp_et_meta_summary       random_5fold          0.9971     0.0016    1.1535     3.1209   1.0000         1.1450
    temp_et_meta_svd       random_5fold          0.9961     0.0021    1.6384     3.9936   1.0000        -1.2727
       temp_mlp_meta       random_5fold          0.9709     0.0156   12.8043    31.7834   0.9978         1.3547
  temp_adaboost_meta       random_5fold          0.8902     0.0590   56.4744    73.7568   0.9883        -0.8041
        temp_rf_meta regime_group_5fold          0.9910     0.0048    3.3448    14.3315   0.9996        43.0251
        temp_et_meta regime_group_5fold          0.9886     0.0061    4.0730    15.7703   0.9995        43.2117
    temp_et_meta_svd regime_group_5fold          0.9760     0.0129    9.9535    25.5607   0.9986         2.6608
temp_et_meta_summary regime_group_5fold          0.9705     0.0158   12.2617    32.6859   0.9977         3.3071
       temp_mlp_meta regime_group_5fold          0.9610     0.0209   17.0935    40.2582   0.9965         1.2160
  temp_adaboost_meta regime_group_5fold          0.8880     0.0602   55.8240    78.4100   0.9868        -0.7123
```

## Temperature rank stability
```
      group            candidate  avg_rank  std_rank  avg_primary
temperature         temp_rf_meta    1.3333    0.5774       0.9859
temperature         temp_et_meta    1.6667    0.5774       0.9779
temperature     temp_et_meta_svd    4.0000    1.0000       0.9592
temperature temp_et_meta_summary    4.3333    1.5275       0.9557
temperature        temp_mlp_meta    4.3333    1.1547       0.9473
temperature   temp_adaboost_meta    5.3333    1.1547       0.8954
```

## Abundance tournament
```
                          candidate             scheme  primary_metric  mean_crps  mean_mae  mean_rmse  mean_r2  mean_nll_norm
abun_shared_mlp_svd_spec_noise_meta       random_5fold          0.7636     0.1379    0.2432     0.4513   0.8896         0.7210
  abun_sep_et_spec_noise_meta_deriv       random_5fold          0.5542     0.2601    0.4746     0.6288   0.7552         0.6196
             abun_sep_et_spec_noise       random_5fold          0.3696     0.3678    0.7112     0.9020   0.5441         0.9613
        abun_sep_et_spec_noise_meta       random_5fold          0.3524     0.3778    0.7330     0.9278   0.5206         0.9886
     abun_shared_et_spec_noise_meta       random_5fold          0.3048     0.4056    0.7935     0.9818   0.4576         1.0489
       alltarget_et_spec_noise_meta       random_5fold          0.1020     0.5239    1.0684     1.3148   0.1403         1.3309
abun_shared_mlp_svd_spec_noise_meta regime_group_5fold          0.7627     0.1384    0.2425     0.4710   0.8789         0.9541
  abun_sep_et_spec_noise_meta_deriv regime_group_5fold          0.5001     0.2916    0.5464     0.7027   0.7073         0.7284
             abun_sep_et_spec_noise regime_group_5fold          0.2471     0.4392    0.8758     1.0918   0.3861         1.1623
        abun_sep_et_spec_noise_meta regime_group_5fold          0.2127     0.4593    0.9232     1.1465   0.3351         1.2084
     abun_shared_et_spec_noise_meta regime_group_5fold          0.1854     0.4752    0.9627     1.1761   0.2964         1.2353
       alltarget_et_spec_noise_meta regime_group_5fold         -0.0892     0.6355    1.3535     1.6138  -0.2305         1.5206
  abun_sep_et_spec_noise_meta_deriv edge_holdout_15pct          0.3963     0.3590    0.6985     0.8874   0.5896         0.9636
             abun_sep_et_spec_noise edge_holdout_15pct          0.2042     0.4733    0.9464     1.1752   0.3012         1.2321
        abun_sep_et_spec_noise_meta edge_holdout_15pct          0.1908     0.4814    0.9718     1.1938   0.2835         1.2472
     abun_shared_et_spec_noise_meta edge_holdout_15pct          0.1888     0.4827    0.9877     1.1888   0.2934         1.2427
       alltarget_et_spec_noise_meta edge_holdout_15pct         -0.0307     0.6148    1.3128     1.5410  -0.0999         1.4874
abun_shared_mlp_svd_spec_noise_meta edge_holdout_15pct         -0.4751     0.8781    1.3519     7.3186 -28.3445        42.2323
```

## Abundance rank stability
```
    group                           candidate  avg_rank  std_rank  avg_primary
abundance   abun_sep_et_spec_noise_meta_deriv    1.6667    0.5774       0.4835
abundance abun_shared_mlp_svd_spec_noise_meta    2.6667    2.8868       0.3504
abundance              abun_sep_et_spec_noise    2.6667    0.5774       0.2736
abundance         abun_sep_et_spec_noise_meta    3.6667    0.5774       0.2520
abundance      abun_shared_et_spec_noise_meta    4.6667    0.5774       0.2263
abundance        alltarget_et_spec_noise_meta    5.6667    0.5774      -0.0060
```

## Target-specialisation evidence
```
          comparison                           candidate  avg_primary
 best_independent_et   abun_sep_et_spec_noise_meta_deriv       0.4835
         shared_tree      abun_shared_et_spec_noise_meta       0.2263
       shared_neural abun_shared_mlp_svd_spec_noise_meta       0.3504
all_target_benchmark        alltarget_et_spec_noise_meta      -0.0060
```

Interpretation:
- Temperature is clearly a specialist metadata task.
- Abundances do not behave like one clean shared target family; independent tree models remain strong contenders.
- Shared models and the all-target benchmark are useful sanity checks, but they must win on score, not elegance.

## Best-so-far diagnostics
### Best temperature model: `temp_rf_meta`
```
            scheme  primary_metric  mean_rmse  mean_r2  mean_nll_norm
edge_holdout_15pct          0.9671    56.1532   0.9963        12.4259
      random_5fold          0.9995     0.4980   1.0000        -5.7041
regime_group_5fold          0.9910    14.3315   0.9996        43.0251
```

### Best abundance model family: `abun_sep_et_spec_noise_meta_deriv`
```
            scheme  target  primary_metric   rmse     r2  sigma_plugin
      random_5fold log_H2O          0.6079 0.7216 0.8267        0.6543
      random_5fold log_CO2          0.6044 0.6013 0.8268        0.5660
      random_5fold log_CH4          0.7714 0.4434 0.9351        0.3484
      random_5fold  log_CO          0.2723 0.6403 0.4498        0.6715
      random_5fold log_NH3          0.5153 0.7376 0.7377        0.6795
regime_group_5fold log_H2O          0.5687 0.7845 0.7952        0.7407
regime_group_5fold log_CO2          0.5481 0.6792 0.7790        0.6638
regime_group_5fold log_CH4          0.6875 0.5858 0.8868        0.5111
regime_group_5fold  log_CO          0.2148 0.6870 0.3668        0.7346
regime_group_5fold log_NH3          0.4814 0.7772 0.7088        0.7541
edge_holdout_15pct log_H2O          0.4785 0.9702 0.7020        0.9530
edge_holdout_15pct log_CO2          0.4345 0.8738 0.6475        0.8542
edge_holdout_15pct log_CH4          0.5500 0.9008 0.7472        0.7600
edge_holdout_15pct  log_CO          0.1765 0.7228 0.3043        0.7756
edge_holdout_15pct log_NH3          0.3419 0.9694 0.5470        1.0046
```

### Brittleness under shift
```
                          candidate  random_5fold  regime_group_5fold  edge_holdout_15pct  drop_regime  drop_edge  max_drop
abun_shared_mlp_svd_spec_noise_meta        0.7636              0.7627             -0.4751       0.0009     1.2387    1.2387
       alltarget_et_spec_noise_meta        0.1020             -0.0892             -0.0307       0.1912     0.1327    0.1912
             abun_sep_et_spec_noise        0.3696              0.2471              0.2042       0.1225     0.1653    0.1653
        abun_sep_et_spec_noise_meta        0.3524              0.2127              0.1908       0.1397     0.1616    0.1616
  abun_sep_et_spec_noise_meta_deriv        0.5542              0.5001              0.3963       0.0542     0.1579    0.1579
     abun_shared_et_spec_noise_meta        0.3048              0.1854              0.1888       0.1194     0.1160    0.1194
               temp_et_meta_summary        0.9971              0.9705              0.8995       0.0266     0.0975    0.0975
                   temp_et_meta_svd        0.9961              0.9760              0.9055       0.0200     0.0906    0.0906
                      temp_mlp_meta        0.9709              0.9610              0.9099       0.0099     0.0610    0.0610
                       temp_et_meta        0.9995              0.9886              0.9456       0.0110     0.0539    0.0539
                       temp_rf_meta        0.9995              0.9910              0.9671       0.0085     0.0324    0.0324
                 temp_adaboost_meta        0.8902              0.8880              0.9081       0.0022    -0.0178    0.0022
```

What to believe now:
- The model that wins random folds but collapses on regime or edge splits is not competition-ready.
- The better finalist is the one with strong average CRPS proxy and smaller shift-induced drop, not just the best IID score.
- `log_CO` is currently the weakest abundance target under the best abundance family.
