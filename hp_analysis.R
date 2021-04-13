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

d = read_csv("~/Documents/GitHub/softblock/hyperparam_results.csv.gz")

mns = d %>%
    group_by(design, s) %>%
    pivot_wider(names_from=metric) %>%
    summarize(
        time_design=mean(time_design + time_estimation),
        bias_ate=mean(ATEError),
        mse_ate=mean(ATEError^2),
        integrated_bias_ite=mean(ITEBias),
        mise_ite=mean(ITEMSE),
        cov_mise=mean(CovariateMSE)
    ) %>% ungroup() %>%
    pivot_longer(!one_of("design", "s"), names_to="metric")

sems = d %>%
    group_by(design, s) %>%
    pivot_wider(names_from=metric) %>%
    summarize(
        time_design=sd(time_design + time_estimation) / sqrt(n()),
        bias_ate=sd(ATEError) / sqrt(n()),
        mse_ate=sd(ATEError^2) / sqrt(n()),
        integrated_bias_ite=sd(ITEBias) / sqrt(n()),
        mise_ite=sd(ITEMSE) / sqrt(n()),
        cov_mise=sd(CovariateMSE) / sqrt(n())
    ) %>% ungroup() %>%
    pivot_longer(!one_of("design", "s"), names_to="metric", values_to="sem")


log2p1_trans = scales::trans_new('log2p1', transform=function(x) log2(x+1), inverse=function(x) (2^x) - 1)

left_join(mns, sems, by=c("design", "s", "metric")) %>%
    filter(metric %in% c('time_design', "mse_ate", "mise_ite")) %>%
    filter(
        case_when(
            design == "SoftBlock-L" & metric=='mse_ate'~TRUE,
            design == "SoftBlock-RF" & metric=="mise_ite"~TRUE,
            design == "SoftBlock-RF" & metric=="time_design"~TRUE,
            design == "SoftBlock-L"~FALSE,
            design == "SoftBlock-RF"~FALSE,
            TRUE~TRUE,
        ),
    ) %>%
    mutate(
        design=stringr::str_split_fixed(design, '-', 2)[,1],
        metric=case_when(
            metric=="mse_ate"~"MSE(ATE)",
            metric=="mise_ite"~"MISE(CATE)",
            metric=="time_design"~"Time (s)",
            TRUE~"None"
        )
    ) %>%
    ggplot(aes(x = s, y=value)) +
    geom_line(aes(color=design)) +
    geom_ribbon(aes(fill=design, ymin=value-1.96*sem, ymax=value +1.96*sem), alpha=0.5) +
    facet_wrap(~metric, scales='free') +
    scale_y_continuous("", trans=log2p1_trans) +
    scale_x_continuous("Gaussian Kernel Bandwidth", trans='log10') +
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

ggsave("figures/hyperparams.pdf", device=cairo_pdf, width=10, height=3)
