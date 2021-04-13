library(readr)
library(ggplot2)
library(dplyr)
library(tidyr)

df = read_csv('~/Documents/GitHub/SoftBlock/example_data.csv')
L = read_csv('~/Documents/GitHub/SoftBlock/example_laplacian.csv')
df$degree = diag(as.matrix(L))

A = as.matrix(L) - diag(df$degree)
A = as_tibble(ifelse(lower.tri(A),A,0))
names(A) = paste0(1:ncol(A))

ggplot(df, aes(x=X1, y=X2, size=-degree, color=factor(A))) + geom_point(alpha=.75) + theme_minimal()

locs = df %>% mutate(id=as.character(row_number())) %>% select(id, X1, X2)

adj = A %>%
  group_by(id_1=as.character(row_number())) %>%
  gather(id_2, weight, -id_1) %>%
  filter(weight != 0)  %>%
  mutate(id_2=as.character(id_2), id=paste(id_1, id_2, sep='-'))

edges=bind_rows(
adj  %>% inner_join(locs, by=c('id_1'='id')),
adj  %>% inner_join(locs, by=c('id_2'='id'))
) %>% arrange(id) %>% ungroup()

ggplot(df, aes(x=X1, y=X2)) +
geom_point(aes(), alpha=.9, size=5) +
theme_minimal()
ggsave('~/Documents/GitHub/SoftBlock/demo-0.pdf', device=cairo_pdf, width=8, height=8)

ggplot(df, aes(x=X1, y=X2)) +
geom_line(aes(group=id, size=1), data=edges, color='grey') +
geom_point(aes(size=2), alpha=.9) +
scale_size_continuous(range=c(1, 3)) +
theme_minimal() + theme(legend.position='none')
ggsave('~/Documents/GitHub/SoftBlock/demo-1.pdf', device=cairo_pdf, width=8, height=8)

ggplot(df, aes(x=X1, y=X2)) +
geom_line(aes(group=id, size=1), data=edges, color='grey') +
geom_point(aes(color=factor(A), size=2), alpha=.9) +
theme_minimal() +
scale_x_continuous("") +
scale_y_continuous("") +
scale_size_continuous(range=c(1, 3)) +
theme(legend.position='none',
  axis.ticks.x = element_blank(),
  axis.text.x = element_blank(),
  axis.ticks.y = element_blank(),
  axis.text.y = element_blank()
)
ggsave('~/Documents/GitHub/SoftBlock/demo-2.pdf', device=cairo_pdf, width=8, height=8)

ggplot(df, aes(x=X1, y=X2)) +
geom_line(aes(group=id, size=1), data=edges, color='grey') +
geom_point(aes(color=factor(A), size=2), alpha=.9) +
scale_size_continuous(range=c(1, 3)) +
theme_minimal() + theme(legend.position='none')
ggsave('~/Documents/GitHub/SoftBlock/demo-3.pdf', device=cairo_pdf, width=8, height=8)
