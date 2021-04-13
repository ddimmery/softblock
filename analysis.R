library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

design_colors = c(
    "SoftBlock"="#1F77B4FF",
    "GreedyNeighbors"="#FF7F0EFF",
    "KallusHeuristic"="#2CA02CFF",
    "KallusPSOD"="#D62728FF",
    "QuickBlock"="#9467BDFF",
    "OptBlock"="#8C564BFF",
    "Matched Pair"="#E377C2FF",
    "Rerandomization"="#7F7F7FFF",
    "Randomization"="#BCBD22FF"
)

files = dir("results/", pattern=".*gz", full.names=TRUE)

files[!grepl("hyper", files)] %>%
    purrr::map(read_csv, progress=FALSE, col_types = cols()) %>%
    bind_rows -> df

mns = df %>%
    group_by(dgp, design, sample_size) %>%
    pivot_wider(names_from=metric, values_from=value) %>%
    summarize(
        time_design=mean(time_design + time_estimation),
        bias_ate=mean(ATEError),
        mse_ate=mean(ATEError^2),
        integrated_bias_ite=mean(ITEBias),
        mise_ite=mean(ITEMSE),
        cov_mise=mean(CovariateMSE)
    ) %>% ungroup() %>%
    pivot_longer(!one_of("dgp", "design", "sample_size"), names_to="metric")

sems = df %>%
    group_by(dgp, design, sample_size) %>%
    pivot_wider(names_from=metric) %>%
    summarize(
        time_design=sd(time_design + time_estimation) / sqrt(n()),
        bias_ate=sd(ATEError) / sqrt(n()),
        mse_ate=sd(ATEError^2) / sqrt(n()),
        integrated_bias_ite=sd(ITEBias) / sqrt(n()),
        mise_ite=sd(ITEMSE) / sqrt(n()),
        cov_mise=sd(CovariateMSE) / sqrt(n())
    ) %>% ungroup() %>%
    pivot_longer(!one_of("dgp", "design", "sample_size"), names_to="metric", values_to="sem")


log2p1_trans = function() {scales::trans_new(
    'log2p1',
    transform=function(x) log10(x+1),
    inverse=function(x) (10^x) - 1
)}
asinh_trans <- function(){
  scales::trans_new(name = 'asinh', transform = function(x) asinh(x),
            inverse = function(x) sinh(x))
}

pdat = left_join(mns, sems, by=c("dgp", "design", "sample_size", "metric")) %>%
    filter(
        !grepl('IHDP', dgp),
        metric %in% c("mse_ate", "mise_ite")
    ) %>%
    mutate(
        design=ifelse(design=='Randomization', 'Randomization-RF', design),
        design=ifelse(design=='Matched Pair', 'Matched Pair-B', design),
        old_design=design,
        estimator=stringr::str_split_fixed(design, '-', 2)[,2],
        design=stringr::str_split_fixed(design, '-', 2)[,1]
    ) %>%
    filter(
        case_when(
            old_design == "SoftBlock-L" & metric=='mse_ate'~TRUE,
            old_design == "SoftBlock-RF" & metric=="mise_ite"~TRUE,
            old_design == "SoftBlock-RF" & metric=="time_design"~TRUE,
            old_design == "GreedyNeighbors-L" & metric=='mse_ate'~TRUE,
            old_design == "GreedyNeighbors-RF" & metric=="mise_ite"~TRUE,
            old_design == "GreedyNeighbors-RF" & metric=="time_design"~TRUE,
            design != "GreedyNeighbors" & design != "SoftBlock" & estimator=="RF"~TRUE,
            TRUE~FALSE
        ),
        design!='Randomization',
        design!='Fixed Margins Randomization',
        design!='OptBlock',
        design!='KallusHeuristic',
        sample_size < 1e6
    ) %>%
    ungroup() %>%
    mutate(
        n_for_norm=case_when(
            metric=="mse_ate"~sample_size,
            metric=="mise_ite" & dgp %in% c("QuickBlockDGP", "TwoCircles")~sqrt(sample_size),
            metric=="mise_ite" & dgp %in% c("LinearDGP", "SinusoidalDGP")~(sample_size)^0.25,
            TRUE~sample_size
        ),
        metric=case_when(
            metric=="mse_ate"~"MSE(ATE) × n",
            metric=="mise_ite" & dgp %in% c("QuickBlockDGP", "TwoCircles")~"MISE(CATE) × √n",
            metric=="mise_ite"~"MISE(CATE) × n^¼",
            TRUE~"None"
        )
    )

p_main = pdat %>%
    ggplot(aes(x = sample_size, y=value * n_for_norm, group=old_design)) +
    geom_ribbon(aes(fill=design, ymin=(value-1.6*sem) * n_for_norm, ymax=(value +1.6*sem)* n_for_norm), alpha=0.25) +
    geom_line(aes(color=design)) +
    scale_y_continuous("", trans='log10', labels=scales::comma) +
    scale_x_continuous("Sample Size", trans='log10', breaks = c(10, 100, 1000, 10000, 100000, 1000000),
              labels = scales::trans_format("log10", scales::math_format(10^.x))) +
    scale_fill_manual("", values = design_colors) +
    scale_color_manual("", values = design_colors) +
    facet_wrap(dgp~metric, scales='free_y', ncol=2) +
    theme_minimal() +
    theme(
        legend.position='bottom',
        panel.border=element_rect(color='black', size=0.5, fill=NA),
        axis.title.x=element_text(size=14),
        axis.title.y=element_text(size=14),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        strip.text.x=element_text(size=12),
    )

print(p_main)
ggsave("figures/main.pdf", device=cairo_pdf, width=7, height=7)

p_design = left_join(mns, sems, by=c("dgp", "design", "sample_size", "metric")) %>%
    filter(
        !grepl('IHDP', dgp),
        metric %in% c("mse_ate", "mise_ite")
    ) %>%
    mutate(
        design=ifelse(design=='Randomization', 'Randomization-RF', design),
        design=ifelse(design=='Matched Pair', 'Matched Pair-B', design),
        old_design=design,
        estimator=stringr::str_split_fixed(design, '-', 2)[,2],
        design=stringr::str_split_fixed(design, '-', 2)[,1]
    ) %>%
    filter(
        estimator %in% c('L', 'B'),
        design!='OptBlock',
        sample_size < 1e6
    ) %>%
    ungroup() %>%
    mutate(
        n_for_norm=case_when(
            metric=="mse_ate"~sample_size,
            metric=="mise_ite" & dgp %in% c("QuickBlockDGP", "TwoCircles")~sqrt(sample_size),
            metric=="mise_ite" & dgp %in% c("LinearDGP", "SinusoidalDGP")~(sample_size)^0.25,
            TRUE~sample_size
        ),
        metric=case_when(
            metric=="mse_ate"~"MSE(ATE) × n",
            metric=="mise_ite"~"MISE(CATE) × n^1/K",
            TRUE~"None"
        )
    ) %>%
    ggplot(aes(x = sample_size, y=value * n_for_norm, group=old_design)) +
    geom_ribbon(aes(fill=design, ymin=(value-1.6*sem) * n_for_norm, ymax=(value +1.6*sem) * n_for_norm), alpha=0.25) +
    geom_line(aes(color=design)) +
    scale_y_continuous("", trans='log10', breaks = scales::trans_breaks("log10", function(x) 10^x),
              labels = scales::trans_format("log10", scales::math_format(10^.x))) +
    scale_x_continuous("Sample Size", trans='log10', breaks = c(10, 100, 1000, 10000, 100000, 1000000),
              labels = scales::trans_format("log10", scales::math_format(10^.x))) +
    scale_fill_manual("", values = design_colors) +
    scale_color_manual("", values = design_colors) +
    geom_hline(yintercept=0, linetype='dashed', color='black', alpha=0.5) +
    facet_grid(metric~dgp, scale='free') +
    theme_minimal() +
    theme(
        legend.position='bottom',
        panel.border=element_rect(color='black', size=0.5, fill=NA),
        axis.title.x=element_text(size=14),
        axis.title.y=element_text(size=14),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        strip.text.x=element_text(size=12),
    )

print(p_design)
ggsave("figures/designs.pdf", device=cairo_pdf, width=7, height=5.5)

p_time = left_join(mns, sems, by=c("dgp", "design", "sample_size", "metric")) %>%
    filter(
        !grepl('IHDP', dgp),
        metric %in% c('time_design')
    ) %>%
    mutate(
        design=ifelse(design=='Randomization', 'Randomization-RF', design),
        design=ifelse(design=='Matched Pair', 'Matched Pair-B', design),
        old_design=design,
        estimator=stringr::str_split_fixed(design, '-', 2)[,2],
        design=stringr::str_split_fixed(design, '-', 2)[,1]
    ) %>%
    group_by(dgp, metric, sample_size) %>%
    filter(
        design!='Randomization',
        dgp=='TwoCircles',
        estimator=='RF',
        design!="Fixed Margins Randomization",
        design!="OptBlock"
    ) %>%
    ggplot(aes(x = sample_size, y=value, group=old_design)) +
    geom_ribbon(aes(fill=design, ymin=value-1.6*sem, ymax=value +1.6*sem), alpha=0.25) +
    geom_line(aes(color=design)) +
    scale_y_continuous('Time (s)', trans='log10') +
    scale_x_continuous("Sample Size", trans='log10', breaks = c(10, 100, 1000, 10000, 100000, 1000000),
              labels = scales::trans_format("log10", scales::math_format(10^.x))) +
    scale_fill_manual("", values = design_colors) +
    scale_color_manual("", values = design_colors) +
    theme_minimal() +
    theme(
        legend.position='bottom',
        panel.border=element_rect(color='black', size=0.5, fill=NA),
        axis.title.x=element_text(size=14),
        axis.title.y=element_text(size=14),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        strip.text.x=element_text(size=12),
    )
print(p_time)
ggsave("figures/runtime.pdf", device=cairo_pdf, width=6.5, height=3)

p_ihdp = left_join(mns, sems, by=c("dgp", "design", "sample_size", "metric")) %>%
    filter(
        grepl('IHDP', dgp),
        metric %in% c('time_design', "mse_ate", "mise_ite")
    ) %>%
    mutate(
        design=ifelse(design=='Randomization', 'Randomization-RF', design),
        design=ifelse(design=='Matched Pair', 'Matched Pair-B', design),
        old_design=design,
        estimator=stringr::str_split_fixed(design, '-', 2)[,2],
        design=stringr::str_split_fixed(design, '-', 2)[,1]
    ) %>%
    filter(
        estimator == 'RF',
        design!='Fixed Margins Randomization',
    ) %>%
    ungroup() %>%
    mutate(metric=case_when(
            metric=="mise_ite"~"MISE(CATE)",
            metric=="mse_ate"~"MSE(ATE)",
            metric=="time_design"~"Time (s)",
            TRUE~"None",
    )
        ) %>%
    filter(sample_size==0) %>%
    ggplot(aes(x = design, y=value, group=old_design)) +
    geom_pointrange(aes(color=design, ymin=value-1.6*sem, ymax=value +1.6*sem)) +
    scale_y_continuous("", trans='log10') +
    scale_x_discrete("") +
    scale_fill_manual("", values = design_colors) +
    scale_color_manual("", values = design_colors) +
    facet_wrap(~metric, scale='free_x') +
    coord_flip() +
    theme_minimal() +
    theme(
        legend.position='none',
        panel.border=element_rect(color='black', size=0.5, fill=NA),
        axis.title.x=element_text(size=14),
        axis.title.y=element_text(size=14),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        strip.text.x=element_text(size=12),
    )

print(p_ihdp)

ggsave("figures/ihdp.pdf", device=cairo_pdf, width=8, height=2)
