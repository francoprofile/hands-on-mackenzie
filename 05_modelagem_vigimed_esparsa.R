# =============================================================================
# VigiMed (Anvisa) | Modelagem de `ind_grave` com matrizes esparsas
# -----------------------------------------------------------------------------
# Modelos : XGBoost | Regressão logística Elastic Net (glmnet) | Random Forest (ranger)
# Alvo    : ind_grave (1 = Grave = evento de interesse; ~54% graves, balanceado)
#
# COMO USAR
#  1. Ajuste os parâmetros e rode o BLOCO 0 (em toda sessão nova do R).
#  2. Rode os blocos 1 a 7 em ordem. Cada bloco lê do disco o que precisa, grava o
#     que produz em DIR_ARTEFATOS e remove da memória o que não vai mais usar:
#     dá para reiniciar o R entre blocos.
#  3. Antes da execução completa, rode tudo com PCT_AMOSTRA <- 0.05 (5% das linhas)
#     para validar o pipeline em poucos minutos. Os artefatos da amostra vão para
#     outra pasta e não se misturam com os da execução completa.
#
# FLUXO E ARTEFATOS (pasta artefatos/)
#  Bloco 1  csv  -> 01_treino.rds, 01_teste.rds             data.frames compactos
#  Bloco 2  01_* -> 02_treino_esparso.rds, 02_teste_esparso.rds   dgCMatrix one-hot
#  Bloco 3  02_treino_esparso.rds -> modelo_<nome>.rds, resultado_<nome>.rds
#  Bloco 4  02_teste_esparso.rds + modelos -> comparacao_modelos.csv, 04_predicoes_teste.rds
#  Bloco 5  modelos + teste -> importancia_variaveis.png / .csv
#  Bloco 6  04_predicoes_teste.rds -> matriz_confusao.csv
#  Bloco 7  04_predicoes_teste.rds -> curva_roc.png
#
# DECISÕES VINDAS DA EDA (04_eda_vigimed.R)
#  - Alvo 54/46: balanceado, sem class_weight/SMOTE.
#  - Fora do modelo: id_notificacao (identificador); origem_ind_grave (auditoria do
#    rótulo = vazamento); posologia (texto livre, ~168 mil valores); nome_medicamento
#    (redundante com principios_ativos); mes_inclusao_sistema (p = 0,15); níveis
#    MedDRA colineares (fica só reacao_soc).
#  - desfecho: "Fatal/Óbito" tem 98% de graves e óbito é critério de gravidade, então
#    fica fora por padrão (USAR_DESFECHO). Ligue só se o campo existir quando o
#    modelo for usado na prática.
#  - Idade 49 = mediana que a Gold usa para nulos (11,7% das linhas) -> flag
#    idade_imputada. atc_nivel1 (1º caractere) no lugar de codigo_atc.
# =============================================================================


# =============================================================================
# BLOCO 0 - PARÂMETROS E FUNÇÕES COMUNS (rodar em toda sessão nova)
# =============================================================================

pacotes  <- c("dplyr", "Matrix", "xgboost", "glmnet", "ranger", "doParallel", "foreach", "ggplot2")
faltando <- setdiff(pacotes, rownames(installed.packages()))
if (length(faltando) > 0) {
  stop("Instale antes: install.packages(c(", paste0('"', faltando, '"', collapse = ", "), "))")
}
library(dplyr)
library(Matrix)
library(doParallel)   # já carrega o foreach (operador %dopar%)
library(ggplot2)      # gráficos dos blocos 5 e 7

# Carrega (sem anexar) os pacotes dos modelos: o predict() de um modelo lido do disco
# só funciona com o pacote que o criou carregado (vale ao reiniciar o R entre blocos).
invisible(lapply(c("xgboost", "glmnet", "ranger"), loadNamespace))

# ---- Parâmetros --------------------------------------------------------------
ARQ_CSV       <- "anvisa.g_vigimed_obt_ml.csv"
SEMENTE       <- 42
PCT_AMOSTRA   <- 1       # 1 = base toda | 0.05 = 5% das linhas (teste rápido do pipeline)
PROP_TREINO   <- 0.80
N_TOP         <- 30      # categorias mantidas por variável de alta cardinalidade
V_FOLDS       <- 3       # validação cruzada leve
CORTE         <- 0.5     # corte de probabilidade para F1, sensibilidade etc. (54/46 -> 0,5)
USAR_DESFECHO <- FALSE   # TRUE inclui desfecho (risco de vazamento, ver cabeçalho)

# Limites dos grids (grids enxutos: poucas combinações, escolhidas pelo ROC-AUC)
NROUNDS_MAX   <- 500     # XGBoost: máximo de árvores (early stopping corta antes)
ARVORES_TUNE  <- 100     # Random Forest: árvores durante o tuning
ARVORES_FINAL <- 300     # Random Forest: árvores no modelo final
N_TUNE_RF     <- 80000   # Random Forest: linhas usadas no tuning (acelera e poupa RAM)

# Cada worker é um R separado com cópia dos dados: no máximo 4 (núcleos físicos - 1)
N_WORKERS <- max(1, min(4, parallel::detectCores(logical = FALSE) - 1, na.rm = TRUE))

# Colunas que NÃO entram no modelo (motivos no cabeçalho). Nem são lidas do CSV.
VARS_FORA <- c("id_notificacao", "origem_ind_grave", "posologia", "nome_medicamento",
               "mes_inclusao_sistema", "reacao_llt", "reacao_pt", "reacao_hlt", "reacao_hlgt")
if (!USAR_DESFECHO) VARS_FORA <- c(VARS_FORA, "desfecho")

# Rótulos e cores dos modelos nos gráficos (mesma ordem em todos os blocos)
ROTULO_MODELOS <- c(xgboost = "XGBoost", elasticnet = "Elastic Net", random_forest = "Random Forest")
COR_MODELOS    <- c(xgboost = "#2a78d6", elasticnet = "#eb6834", random_forest = "#2a9d8f")

DIR_ARTEFATOS <- if (PCT_AMOSTRA < 1) "artefatos_amostra" else "artefatos"
dir.create(DIR_ARTEFATOS, showWarnings = FALSE)
caminho <- function(arquivo) file.path(DIR_ARTEFATOS, arquivo)

# ---- Métricas ----------------------------------------------------------------
# AUC pelo método dos postos (equivale ao teste de Mann-Whitney): sem pacotes extras.
# y = 0/1 (1 = Grave); p = probabilidade prevista de Grave.
auc_rank <- function(y, p) {
  ok <- !is.na(p)
  y  <- y[ok]
  r  <- rank(p[ok])
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# ROC-AUC (independe do corte) + métricas de classe no corte escolhido.
# "Grave" é a classe positiva: sensibilidade = recall dos graves.
calcula_metricas <- function(y, p, corte = 0.5) {
  pred <- as.integer(p >= corte)
  tp <- sum(pred == 1 & y == 1)
  fp <- sum(pred == 1 & y == 0)
  fn <- sum(pred == 0 & y == 1)
  tn <- sum(pred == 0 & y == 0)
  precisao <- tp / (tp + fp)
  recall   <- tp / (tp + fn)
  c(roc_auc              = auc_rank(y, p),
    f1                   = 2 * precisao * recall / (precisao + recall),
    sensibilidade_recall = recall,
    especificidade       = tn / (tn + fp),
    precisao             = precisao)
}


# =============================================================================
# BLOCO 1 - PREPARAÇÃO E FEATURE ENGINEERING
# Função : lê o CSV, cria as variáveis derivadas, divide em treino/teste e agrupa
#          as categorias raras.
# Motivo : categóricas com centenas de níveis (princípio ativo, empresa...) gerariam
#          centenas de colunas dummy quase vazias. Mantemos as N_TOP mais frequentes
#          e agrupamos o resto em "Outros". O ranking é calculado SÓ no treino (o
#          teste não influencia nada) e depois aplicado ao teste.
# Saída  : 01_treino.rds e 01_teste.rds (factors com níveis fixos = top N + "Outros").
#          Esses níveis definem as colunas dummy; a expansão one-hot acontece no
#          Bloco 2 já em formato esparso, para que a versão densa nunca exista na RAM.
# =============================================================================

set.seed(SEMENTE)

# ---- 1.1 Leitura apenas das colunas necessárias -------------------------------
# colClasses = "NULL" faz o read.csv pular a coluna: textos longos (posologia,
# reacao_llt...) nem chegam à memória.
cab   <- names(read.csv(ARQ_CSV, nrows = 1))
tipos <- ifelse(cab %in% VARS_FORA, "NULL", NA)
dados <- read.csv(ARQ_CSV, colClasses = tipos, stringsAsFactors = FALSE)

if (PCT_AMOSTRA < 1) {   # amostra para teste rápido do pipeline
  dados <- dados[sample.int(nrow(dados), round(PCT_AMOSTRA * nrow(dados))), ]
}
stopifnot(!anyNA(dados), all(dados$ind_grave %in% 0:1))   # a Gold promete zero nulos

# ---- 1.2 Variáveis derivadas --------------------------------------------------
# Só regras fixas (sem aprender nada dos dados), por isso podem vir antes da divisão.
dados <- dados %>%
  mutate(
    # grupo anatômico do ATC (1º caractere), como na EDA
    atc_nivel1 = ifelse(codigo_atc == "Nao Informado", "Nao Informado", substr(codigo_atc, 1, 1)),
    # 49 = mediana que a Gold usou para preencher idades nulas
    idade_imputada = as.integer(idade_momento_reacao == 49),
    # faixas da EDA: captura a relação não linear da idade (ajuda sobretudo o Elastic Net)
    faixa_etaria = cut(idade_momento_reacao, right = FALSE,
                       breaks = c(-Inf, 2, 12, 18, 40, 60, 80, Inf),
                       labels = c("0-1", "2-11", "12-17", "18-39", "40-59", "60-79", "80+")),
    # contagens muito assimétricas (máx. 135): log1p ajuda o modelo linear e não muda as árvores
    log_qtd_reacoes      = log1p(qtd_reacoes),
    log_qtd_medicamentos = log1p(qtd_medicamentos),
    # ano é calendário, não quantidade: vira categoria
    ano_inclusao_sistema = as.character(ano_inclusao_sistema)
  ) %>%
  select(-codigo_atc, -qtd_reacoes, -qtd_medicamentos)

# ---- 1.3 Divisão treino/teste estratificada pelo alvo -------------------------
# O teste só é usado na avaliação final (Bloco 4).
idx_tr <- unlist(lapply(split(seq_len(nrow(dados)), dados$ind_grave),
                        function(i) i[sample.int(length(i), round(PROP_TREINO * length(i)))]),
                 use.names = FALSE)
idx_tr <- sample(idx_tr)   # embaralha (o unlist deixa as classes em blocos)
treino <- dados[idx_tr, ]
teste  <- dados[-idx_tr, ]
rm(dados, idx_tr); invisible(gc())

cat(sprintf("Treino: %d linhas (%.1f%% graves) | Teste: %d linhas (%.1f%% graves)\n",
            nrow(treino), 100 * mean(treino$ind_grave),
            nrow(teste),  100 * mean(teste$ind_grave)))

# ---- 1.4 Agrupamento das categorias raras (alta cardinalidade) ----------------
# Texto vira factor com níveis FIXOS: top N do treino + "Outros". Níveis que só
# aparecem no teste caem em "Outros", então treino e teste ficam sempre alinhados.
agrupa <- function(x, top) factor(ifelse(x %in% top, x, "Outros"),
                                  levels = c(setdiff(top, "Outros"), "Outros"))

vars_texto <- names(treino)[sapply(treino, is.character)]
for (v in vars_texto) {
  top <- head(names(sort(table(treino[[v]]), decreasing = TRUE)), N_TOP)
  treino[[v]] <- agrupa(treino[[v]], top)
  teste[[v]]  <- agrupa(teste[[v]],  top)
}

# Conferência: % das linhas de treino em "Outros" (quanto maior, mais informação agrupada)
print(sapply(treino[vars_texto], function(x) round(100 * mean(x == "Outros"), 1)))

saveRDS(treino, caminho("01_treino.rds"))
saveRDS(teste,  caminho("01_teste.rds"))
rm(treino, teste, vars_texto, cab, tipos, v, top); invisible(gc())


# =============================================================================
# BLOCO 2 - CONVERSÃO PARA MATRIZ ESPARSA
# Função : transforma os data.frames em matrizes esparsas (dgCMatrix) com one-hot.
# Motivo : cada linha tem ~20 valores não nulos em ~250 colunas (densidade < 10%).
#          Densa, a matriz ocuparia ~10x mais RAM, e ainda seria copiada para cada
#          worker do doParallel. A esparsa é aceita direto por glmnet, ranger e
#          (via xgb.DMatrix) xgboost, sem nunca virar matriz densa.
# Saída  : 02_treino_esparso.rds (X, y, dobras da CV) e 02_teste_esparso.rds (X, y).
# =============================================================================

set.seed(SEMENTE)

# One-hot COMPLETO (todas as categorias viram coluna) e direto em esparsa.
# Numéricas e flags 0/1 passam como estão.
codifica <- function(df) {
  fatores <- names(df)[sapply(df, is.factor)]
  sparse.model.matrix(~ . - 1, data = df,
                      contrasts.arg = lapply(df[fatores], contrasts, contrasts = FALSE),
                      row.names = FALSE)
}

# ---- 2.1 Dummificação esparsa (um conjunto por vez, para limitar o pico de RAM) -
treino <- readRDS(caminho("01_treino.rds"))
y_tr   <- treino$ind_grave
X_tr   <- codifica(treino[names(treino) != "ind_grave"])
rm(treino); invisible(gc())

teste <- readRDS(caminho("01_teste.rds"))
y_te  <- teste$ind_grave
X_te  <- codifica(teste[names(teste) != "ind_grave"])
rm(teste); invisible(gc())

# ---- 2.2 Limpeza das colunas ---------------------------------------------------
# (a) treino e teste devem ter exatamente as mesmas colunas;
# (b) remove colunas sem nenhum valor no treino (ex.: "Outros" vazio quando todas as
#     categorias couberam no top N); diff(p) = nº de valores não nulos por coluna;
# (c) nomes seguros: o xgboost recusa colchetes e "<" nos nomes das colunas.
stopifnot(identical(colnames(X_tr), colnames(X_te)),
          nrow(X_tr) == length(y_tr), nrow(X_te) == length(y_te))
mantem <- diff(X_tr@p) > 0
X_tr   <- X_tr[, mantem, drop = FALSE]
X_te   <- X_te[, mantem, drop = FALSE]
colnames(X_tr) <- make.names(colnames(X_tr), unique = TRUE)
colnames(X_te) <- colnames(X_tr)

# ---- 2.3 Dobras estratificadas da validação cruzada ---------------------------
# Criadas uma vez: XGBoost e Elastic Net são comparados nas MESMAS dobras.
foldid <- integer(length(y_tr))
for (classe in 0:1) {
  i <- which(y_tr == classe)
  foldid[i] <- sample(rep_len(seq_len(V_FOLDS), length(i)))
}

# ---- 2.4 Teste de conversão para xgb.DMatrix ----------------------------------
# O DMatrix é um ponteiro C++ (não vai para RDS nem para os workers): é montado a
# partir da dgCMatrix onde for usado. Testar aqui evita descobrir o erro no tuning.
dtr <- xgboost::xgb.DMatrix(X_tr, label = y_tr)
cat("xgb.DMatrix OK:", nrow(dtr), "linhas x", ncol(dtr), "colunas\n")
rm(dtr)

# ---- 2.5 Relatório de memória: esparsa x densa --------------------------------
cat(sprintf("Treino: %d x %d | densidade %.1f%% | esparsa %.0f MB vs densa ~%.0f MB\n",
            nrow(X_tr), ncol(X_tr), 100 * nnzero(X_tr) / prod(dim(X_tr)),
            as.numeric(object.size(X_tr)) / 1e6, prod(dim(X_tr)) * 8 / 1e6))

saveRDS(list(X_tr = X_tr, y_tr = y_tr, foldid = foldid), caminho("02_treino_esparso.rds"))
saveRDS(list(X_te = X_te, y_te = y_te),                  caminho("02_teste_esparso.rds"))
rm(X_tr, X_te, y_tr, y_te, foldid, mantem, classe, i, codifica); invisible(gc())


# =============================================================================
# BLOCO 3 - TREINAMENTO E TUNING (doParallel)
# Função : para cada modelo, escolhe os hiperparâmetros por ROC-AUC (validação
#          cruzada, ou OOB no Random Forest), ajusta o modelo final no treino
#          inteiro e salva modelo + resultado do tuning em disco.
# Motivo : o paralelismo (doParallel) roda as combinações do grid em workers
#          separados; cada worker usa 1 thread, então não há disputa de CPU. Ao
#          terminar cada modelo, ele é gravado, removido da RAM e o gc() é chamado.
# Saída  : modelo_<nome>.rds e resultado_<nome>.rds (nome = xgboost | elasticnet |
#          random_forest).
# =============================================================================

set.seed(SEMENTE)

d      <- readRDS(caminho("02_treino_esparso.rds"))
X_tr   <- d$X_tr
y_tr   <- d$y_tr
foldid <- d$foldid
rm(d)
folds_cv <- split(seq_along(foldid), foldid)   # índices de cada dobra (formato do xgb.cv)

# ---- 3.1 Cluster de workers ---------------------------------------------------
cl <- parallel::makeCluster(N_WORKERS)
registerDoParallel(cl)

# O tryCatch(finally) garante que o cluster seja encerrado mesmo se algo falhar no
# meio (senão os workers ficariam ocupando RAM até fechar o R).
tryCatch({

  # ---- 3.2 XGBoost ------------------------------------------------------------
  # Grid: 3 profundidades x 3 taxas de aprendizado = 9 combinações, cada uma avaliada
  # com xgb.cv em um worker. O early stopping define o nº de árvores de cada combinação.
  t0 <- Sys.time()

  params_fixos <- list(objective = "binary:logistic", eval_metric = "auc",
                       tree_method = "hist",   # algoritmo rápido e econômico em CPU
                       subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 5,
                       seed = SEMENTE)
  grade_xgb <- expand.grid(max_depth = c(4, 6, 8), eta = c(0.05, 0.1, 0.2))

  res_xgb <- foreach(i = seq_len(nrow(grade_xgb)), .combine = rbind,
                     .packages = c("xgboost", "Matrix")) %dopar% {
    dtr <- xgboost::xgb.DMatrix(X_tr, label = y_tr, nthread = 1)
    prm <- c(params_fixos, list(max_depth = grade_xgb$max_depth[i],
                                eta = grade_xgb$eta[i], nthread = 1))
    cv  <- xgboost::xgb.cv(params = prm, data = dtr, nrounds = NROUNDS_MAX, nfold = V_FOLDS,
                           folds = folds_cv, early_stopping_rounds = 20,
                           maximize = TRUE, verbose = FALSE)
    aucs <- as.data.frame(cv$evaluation_log)$test_auc_mean
    data.frame(max_depth = grade_xgb$max_depth[i], eta = grade_xgb$eta[i],
               nrounds = which.max(aucs), auc_cv = max(aucs))
  }
  print(res_xgb[order(-res_xgb$auc_cv), ], row.names = FALSE)
  melhor <- res_xgb[which.max(res_xgb$auc_cv), ]
  invisible(parallel::clusterEvalQ(cl, gc()))   # workers devolvem a RAM da validação cruzada

  # Ajuste final no treino inteiro, agora com todos os núcleos (multi-thread nativo)
  dtrain <- xgboost::xgb.DMatrix(X_tr, label = y_tr)
  modelo_xgb <- xgboost::xgb.train(
    params  = c(params_fixos, list(max_depth = melhor$max_depth, eta = melhor$eta,
                                   nthread = N_WORKERS)),
    data    = dtrain, nrounds = melhor$nrounds, verbose = 0)

  tempo <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  saveRDS(modelo_xgb, caminho("modelo_xgboost.rds"))
  saveRDS(list(modelo = "xgboost", melhor = melhor, tuning = res_xgb,
               auc_validacao = melhor$auc_cv, tempo_min = tempo),
          caminho("resultado_xgboost.rds"))
  message(sprintf("XGBoost: %.1f min | AUC na CV = %.4f", tempo, melhor$auc_cv))
  rm(modelo_xgb, dtrain, res_xgb, melhor, grade_xgb, params_fixos); invisible(gc())


  # ---- 3.3 Regressão logística Elastic Net (glmnet) ---------------------------
  # Grid: alpha (0 = ridge ... 1 = lasso). O lambda NÃO precisa de grid: o glmnet
  # percorre um caminho de ~100 valores e o cv.glmnet escolhe o de maior AUC.
  # Uma combinação (alpha) por worker. O glmnet padroniza as colunas internamente,
  # sem tornar a matriz densa.
  t0 <- Sys.time()

  alphas <- c(0, 0.25, 0.5, 0.75, 1)
  res_net <- foreach(a = alphas, .combine = rbind, .packages = c("glmnet", "Matrix")) %dopar% {
    cv <- glmnet::cv.glmnet(X_tr, y_tr, family = "binomial", alpha = a,
                            type.measure = "auc", foldid = foldid)
    i  <- which.max(cv$cvm)
    data.frame(alpha = a, lambda = cv$lambda[i], auc_cv = cv$cvm[i])
  }
  print(res_net[order(-res_net$auc_cv), ], row.names = FALSE)
  melhor <- res_net[which.max(res_net$auc_cv), ]
  invisible(parallel::clusterEvalQ(cl, gc()))

  # Modelo final: refaz o cv.glmnet só com o melhor alpha (as 3 dobras em paralelo).
  # O objeto guarda o caminho de lambdas; o Bloco 4 usa s = "lambda.min".
  modelo_net <- glmnet::cv.glmnet(X_tr, y_tr, family = "binomial", alpha = melhor$alpha,
                                  type.measure = "auc", foldid = foldid, parallel = TRUE)

  tempo <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  saveRDS(modelo_net, caminho("modelo_elasticnet.rds"))
  saveRDS(list(modelo = "elasticnet", melhor = melhor, tuning = res_net,
               auc_validacao = melhor$auc_cv, tempo_min = tempo),
          caminho("resultado_elasticnet.rds"))
  message(sprintf("Elastic Net: %.1f min | AUC na CV = %.4f", tempo, melhor$auc_cv))
  rm(modelo_net, res_net, melhor, alphas); invisible(gc())


  # ---- 3.4 Random Forest (ranger) ---------------------------------------------
  # O ranger aceita dgCMatrix, desde que o alvo seja uma coluna da própria matriz.
  # Alvo 0/1 numérico = floresta de regressão: com alvo binário o critério de divisão
  # (redução de variância) equivale ao Gini, e a média das folhas é a probabilidade de
  # Grave (mesma ideia da "probability forest"). Evita ambiguidade nas colunas de
  # probabilidade.
  # Otimização de RAM/tempo:
  #  - tuning com OOB (sem validação cruzada: 1 ajuste por combinação) numa subamostra;
  #  - sample.fraction = 0.5 sem reposição: cada árvore vê metade das linhas
  #    (árvores menores e mais decorrelacionadas);
  #  - min.node.size alto limita o tamanho das árvores;
  #  - write.forest = FALSE no tuning (só o modelo final guarda as árvores).
  t0 <- Sys.time()

  rf_tr <- cbind(ind_grave = y_tr, X_tr)
  colnames(rf_tr)[1] <- "ind_grave"
  stopifnot(inherits(rf_tr, "dgCMatrix"))

  idx_tune <- sample.int(nrow(rf_tr), min(N_TUNE_RF, nrow(rf_tr)))
  rf_tune  <- rf_tr[idx_tune, ]
  y_tune   <- y_tr[idx_tune]

  # mtry proporcional ao nº de colunas (5%, 10% e 20%) x 2 tamanhos mínimos de nó
  grade_rf <- expand.grid(mtry  = unique(pmax(1, round(ncol(X_tr) * c(0.05, 0.10, 0.20)))),
                          min_n = c(30, 100))

  res_rf <- foreach(i = seq_len(nrow(grade_rf)), .combine = rbind,
                    .packages = c("ranger", "Matrix")) %dopar% {
    m <- ranger::ranger(dependent.variable.name = "ind_grave", data = rf_tune,
                        num.trees = ARVORES_TUNE, mtry = grade_rf$mtry[i],
                        min.node.size = grade_rf$min_n[i], sample.fraction = 0.5,
                        replace = FALSE, write.forest = FALSE, num.threads = 1,
                        seed = SEMENTE)
    data.frame(mtry = grade_rf$mtry[i], min_n = grade_rf$min_n[i],
               auc_oob = auc_rank(y_tune, m$predictions))
  }
  print(res_rf[order(-res_rf$auc_oob), ], row.names = FALSE)
  melhor <- res_rf[which.max(res_rf$auc_oob), ]
  rm(rf_tune, y_tune, idx_tune); invisible(parallel::clusterEvalQ(cl, gc()))

  # Ajuste final: treino inteiro, mais árvores, todos os núcleos. O AUC "OOB" do
  # modelo final (linhas que cada árvore não viu) faz o papel do AUC da validação cruzada.
  modelo_rf <- ranger::ranger(dependent.variable.name = "ind_grave", data = rf_tr,
                              num.trees = ARVORES_FINAL, mtry = melhor$mtry,
                              min.node.size = melhor$min_n, sample.fraction = 0.5,
                              replace = FALSE, write.forest = TRUE,
                              num.threads = N_WORKERS, seed = SEMENTE)
  auc_oob <- auc_rank(y_tr, modelo_rf$predictions)

  tempo <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  saveRDS(modelo_rf, caminho("modelo_random_forest.rds"))
  saveRDS(list(modelo = "random_forest", melhor = melhor, tuning = res_rf,
               auc_validacao = auc_oob, tempo_min = tempo),
          caminho("resultado_random_forest.rds"))
  message(sprintf("Random Forest: %.1f min | AUC OOB = %.4f", tempo, auc_oob))
  rm(modelo_rf, rf_tr, res_rf, grade_rf, melhor, auc_oob); invisible(gc())

}, finally = {
  # ---- 3.5 Encerramento do cluster (roda sempre, com ou sem erro) -------------
  parallel::stopCluster(cl)
  registerDoSEQ()   # volta o foreach para o modo sequencial
})

rm(X_tr, y_tr, foldid, folds_cv, cl, tempo, t0); invisible(gc())


# =============================================================================
# BLOCO 4 - AVALIAÇÃO E COMPARAÇÃO FINAL (base de TESTE)
# Função : prevê a base de teste com cada modelo e calcula as métricas.
# Motivo : o teste nunca foi usado antes, então estas métricas são a estimativa
#          honesta de desempenho. Um modelo por vez (carrega, prevê, descarta) para
#          manter a RAM baixa; só as probabilidades previstas ficam na memória.
# Saída  : data.frame `comparacao` (uma linha por modelo, ordenado por ROC-AUC) e
#          comparacao_modelos.csv. auc_validacao x roc_auc: diferença grande = sobreajuste.
# =============================================================================

d    <- readRDS(caminho("02_teste_esparso.rds"))
X_te <- d$X_te
y_te <- d$y_te
rm(d)

# ---- 4.1 Previsões: probabilidade de Grave de cada modelo --------------------
prob <- list()

modelo <- readRDS(caminho("modelo_xgboost.rds"))
prob$xgboost <- predict(modelo, xgboost::xgb.DMatrix(X_te))
rm(modelo); invisible(gc())

modelo <- readRDS(caminho("modelo_elasticnet.rds"))
prob$elasticnet <- as.numeric(predict(modelo, newx = X_te, s = "lambda.min", type = "response"))
rm(modelo); invisible(gc())

modelo <- readRDS(caminho("modelo_random_forest.rds"))
prob$random_forest <- predict(modelo, data = X_te, num.threads = N_WORKERS)$predictions
rm(modelo, X_te); invisible(gc())

# ---- 4.2 Métricas lado a lado --------------------------------------------------
metricas_teste <- t(sapply(prob, function(p) calcula_metricas(y_te, p, CORTE)))

# AUC da validação (CV no XGBoost/Elastic Net, OOB no Random Forest) e tempo de treino
resultados <- lapply(names(prob), function(m) readRDS(caminho(paste0("resultado_", m, ".rds"))))

comparacao <- data.frame(modelo        = names(prob),
                         metricas_teste,
                         auc_validacao = sapply(resultados, function(r) r$auc_validacao),
                         tempo_min     = sapply(resultados, function(r) r$tempo_min),
                         row.names     = NULL)
comparacao <- comparacao[order(-comparacao$roc_auc), ]
comparacao[-1] <- round(comparacao[-1], 4)

cat(sprintf("\nMetricas na base de teste (%d linhas, corte = %.2f):\n", length(y_te), CORTE))
print(comparacao, row.names = FALSE)
cat("\nMelhor modelo por ROC-AUC no teste:", comparacao$modelo[1], "\n")

write.csv(comparacao, caminho("comparacao_modelos.csv"), row.names = FALSE)

# Guarda as probabilidades previstas: os blocos 6 e 7 (matriz de confusão e curva ROC)
# partem delas, sem precisar carregar os modelos de novo.
saveRDS(list(prob = prob, y_te = y_te), caminho("04_predicoes_teste.rds"))
rm(prob, metricas_teste, resultados); invisible(gc())


# =============================================================================
# BLOCO 5 - IMPORTÂNCIA DAS VARIÁVEIS (comparação entre os 3 modelos)
# Função : mede quanto o ROC-AUC no teste CAI quando uma variável é embaralhada
#          (importância por permutação) e compara os três modelos num só gráfico.
# Motivo : as importâncias nativas (ganho do XGBoost, coeficientes do glmnet, impureza
#          do ranger) têm escalas e significados diferentes e não se comparam. A
#          permutação é a mesma régua para todos: a unidade é "pontos de AUC perdidos".
#          Cada variável é embaralhada INTEIRA (todas as suas colunas dummy juntas),
#          então o resultado é por variável (ex.: principios_ativos), não por dummy.
# Atenção: variáveis redundantes dividem a importância (se uma é embaralhada, o modelo
#          usa a outra). Por isso as 3 variáveis de idade são tratadas como um grupo.
# Saída  : data.frame `importancia` (variavel, modelo, queda_auc), importancia_variaveis.csv
#          e importancia_variaveis.png.
# =============================================================================

N_IMPORTANCIA  <- 20000   # linhas do teste usadas (o Random Forest é a parte lenta)
N_VARS_GRAFICO <- 15      # variáveis mostradas no gráfico

# ---- 5.1 Subamostra do teste e grupos de colunas ------------------------------
d <- readRDS(caminho("02_teste_esparso.rds"))
set.seed(SEMENTE)
linhas <- sample.int(nrow(d$X_te), min(N_IMPORTANCIA, nrow(d$X_te)))
X <- d$X_te[linhas, ]
y <- d$y_te[linhas]
rm(d, linhas)

# Cada coluna da matriz pertence à variável cujo nome é o início do nome da coluna
# (ex.: "ufSP." -> uf). Os nomes mais longos vêm primeiro para evitar confusão de prefixo.
vars <- setdiff(names(readRDS(caminho("01_teste.rds"))), "ind_grave")
vars <- vars[order(-nchar(vars))]
grupo <- vapply(colnames(X), function(cn) vars[startsWith(cn, vars)][1], character(1))
stopifnot(!anyNA(grupo))
grupo[grupo %in% c("idade_momento_reacao", "faixa_etaria", "idade_imputada")] <- "idade"
grupos <- unique(grupo)

# Embaralha as linhas só das colunas `cols` (a variável inteira) e devolve a matriz
# esparsa com as colunas na ordem original.
embaralha <- function(X, cols) {
  ordem <- sample.int(nrow(X))
  Xp <- cbind(X[, -cols, drop = FALSE], X[ordem, cols, drop = FALSE])
  Xp[, order(c(seq_len(ncol(X))[-cols], cols)), drop = FALSE]
}
stopifnot(identical(colnames(embaralha(X, which(grupo == grupos[1]))), colnames(X)))

# Probabilidade de Grave de cada modelo para uma matriz esparsa
prever <- list(
  xgboost       = function(m, X) predict(m, xgboost::xgb.DMatrix(X)),
  elasticnet    = function(m, X) as.numeric(predict(m, newx = X, s = "lambda.min", type = "response")),
  random_forest = function(m, X) predict(m, data = X, num.threads = N_WORKERS)$predictions
)

# ---- 5.2 Importância por permutação (um modelo por vez, para poupar RAM) ------
importancia <- list()
for (nome in names(prever)) {
  message("Importância das variáveis: ", ROTULO_MODELOS[[nome]])
  modelo   <- readRDS(caminho(paste0("modelo_", nome, ".rds")))
  auc_base <- auc_rank(y, prever[[nome]](modelo, X))
  auc_perm <- vapply(grupos, function(g) {
    set.seed(SEMENTE)   # mesma permutação para os 3 modelos: comparação justa
    auc_rank(y, prever[[nome]](modelo, embaralha(X, which(grupo == g))))
  }, numeric(1))
  importancia[[nome]] <- data.frame(variavel = grupos, modelo = nome,
                                    queda_auc = auc_base - auc_perm)
  rm(modelo); invisible(gc())
}
importancia <- do.call(rbind, importancia)
rownames(importancia) <- NULL
write.csv(importancia, caminho("importancia_variaveis.csv"), row.names = FALSE)

# ---- 5.3 Gráfico: top variáveis (média dos 3 modelos), uma barra por modelo ---
media_vars <- aggregate(queda_auc ~ variavel, data = importancia, FUN = mean)
top        <- head(media_vars$variavel[order(-media_vars$queda_auc)], N_VARS_GRAFICO)

dados_graf <- importancia[importancia$variavel %in% top, ]
dados_graf$variavel <- factor(dados_graf$variavel, levels = rev(top))   # maior no topo
dados_graf$modelo   <- factor(dados_graf$modelo, levels = names(ROTULO_MODELOS))

g_importancia <- ggplot(dados_graf, aes(x = variavel, y = queda_auc, fill = modelo)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = COR_MODELOS, labels = ROTULO_MODELOS) +
  labs(title    = "Importância das variáveis por modelo",
       subtitle = paste0("Queda do ROC-AUC ao embaralhar a variável (", format(nrow(X), big.mark = "."),
                         " linhas do teste); ordenado pela média dos 3 modelos"),
       x = NULL, y = "Queda do ROC-AUC", fill = NULL) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold"),
        legend.position = "bottom")
print(g_importancia)
ggsave(caminho("importancia_variaveis.png"), g_importancia, width = 9, height = 6.5, dpi = 150)

rm(X, y, vars, grupo, grupos, prever, embaralha, media_vars, top, dados_graf, nome,
   auc_base, auc_perm); invisible(gc())


# =============================================================================
# BLOCO 6 - MATRIZ DE CONFUSÃO (os 3 modelos)
# Função : conta acertos e erros de cada modelo no teste, no corte CORTE.
# Leitura: as linhas são a classe REAL e as colunas a classe PREVISTA.
#            real Grave     x previsto_grave     = verdadeiros positivos (VP)
#            real Grave     x previsto_nao_grave = falsos negativos (FN): graves perdidos
#            real Nao_grave x previsto_grave     = falsos positivos (FP)
#            real Nao_grave x previsto_nao_grave = verdadeiros negativos (VN)
# Saída  : data.frame `matriz_confusao` (6 linhas: 2 por modelo) e matriz_confusao.csv.
# =============================================================================

d    <- readRDS(caminho("04_predicoes_teste.rds"))
prob <- d$prob
y_te <- d$y_te
rm(d)

matriz_confusao <- do.call(rbind, lapply(names(prob), function(nome) {
  pred <- as.integer(prob[[nome]] >= CORTE)
  data.frame(modelo             = nome,
             real               = c("Grave", "Nao_grave"),
             previsto_grave     = c(sum(pred == 1 & y_te == 1), sum(pred == 1 & y_te == 0)),
             previsto_nao_grave = c(sum(pred == 0 & y_te == 1), sum(pred == 0 & y_te == 0)))
}))

cat(sprintf("\nMatriz de confusão no teste (corte = %.2f): linhas = real, colunas = previsto\n", CORTE))
print(matriz_confusao, row.names = FALSE)
write.csv(matriz_confusao, caminho("matriz_confusao.csv"), row.names = FALSE)

rm(prob, y_te); invisible(gc())


# =============================================================================
# BLOCO 7 - CURVA ROC (os 3 modelos)
# Função : desenha a curva ROC de cada modelo no teste, com o AUC na legenda.
# Motivo : a curva mostra o desempenho em TODOS os cortes (o F1 e a matriz de
#          confusão mostram só o corte de 0,5). Quanto mais colada ao canto
#          superior esquerdo, melhor; a diagonal tracejada é o acaso (AUC = 0,5).
# Saída  : curva_roc.png (o gráfico também é exibido no RStudio).
# =============================================================================

d    <- readRDS(caminho("04_predicoes_teste.rds"))
prob <- d$prob
y_te <- d$y_te
rm(d)

# Pontos da curva: para cada valor de probabilidade (do maior para o menor), a taxa de
# graves detectados (tpr) e de não graves classificados como graves (fpr).
curva_roc <- function(y, p) {
  o   <- order(p, decreasing = TRUE)
  y   <- y[o]
  p   <- p[o]
  fim <- c(which(diff(p) != 0), length(p))   # último ponto de cada valor de p (trata empates)
  data.frame(fpr = c(0, cumsum(y == 0)[fim] / sum(y == 0)),
             tpr = c(0, cumsum(y == 1)[fim] / sum(y == 1)))
}

# Legenda com o AUC de cada modelo
rotulo <- vapply(names(prob), function(m)
  sprintf("%s (AUC = %.3f)", ROTULO_MODELOS[[m]], auc_rank(y_te, prob[[m]])), character(1))

roc <- do.call(rbind, lapply(names(prob), function(m)
  data.frame(curva_roc(y_te, prob[[m]]), modelo = rotulo[[m]])))
roc$modelo <- factor(roc$modelo, levels = unname(rotulo))

g_roc <- ggplot(roc, aes(x = fpr, y = tpr, colour = modelo)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
  geom_line(linewidth = 0.9) +
  scale_colour_manual(values = setNames(COR_MODELOS[names(prob)], unname(rotulo))) +
  coord_equal() +
  labs(title    = "Curva ROC no conjunto de teste",
       subtitle = "Detecção de notificações graves (classe positiva = Grave)",
       x = "1 - Especificidade (falsos positivos)", y = "Sensibilidade (verdadeiros positivos)",
       colour = NULL) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold"),
        legend.position = "bottom")
print(g_roc)
ggsave(caminho("curva_roc.png"), g_roc, width = 7, height = 7, dpi = 150)

rm(prob, y_te, curva_roc, rotulo, roc); invisible(gc())
