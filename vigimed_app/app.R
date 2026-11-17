# =============================================================================
# VigiMed (Anvisa) | App Shiny para classificar a gravidade (ind_grave)
# -----------------------------------------------------------------------------
# Modelos (treinados no script 06, em matrizes esparsas):
#   XGBoost | Elastic Net (glmnet) | Random Forest (ranger)
#
#
#
# O QUE É LIDO (uma vez, ao abrir o app)
#  modelo_xgboost.rds, modelo_elasticnet.rds, modelo_random_forest.rds
#  01_treino.rds             -> categorias de cada lista e faixa das numéricas
#  02_teste_esparso.rds      -> só os nomes das colunas da matriz esparsa
#  importancia_variaveis.csv -> ranking que define as N variáveis do formulário
#
# COMO O FORMULÁRIO FUNCIONA
#  - Pede sexo e idade (VARS_FIXAS) e as N_CAMPOS variáveis mais importantes (queda
#    média do ROC-AUC por permutação, bloco 5 do script 06). As demais entram com
#    um valor neutro. Sexo oferece só Masculino e Feminino (SO ESTES) e o notificador
#    é ignorado (VARS_IGNORADAS).
#  - As listas são independentes (sem filtro em cascata): cada uma oferece todos os
#    valores da base, então qualquer combinação pode ser montada.
#
# POR QUE NÃO É UM predict(modelo, data.frame) DIRETO
#  Os modelos aprenderam numa matriz esparsa one-hot, não num data.frame. Por isso
#  o app monta o data.frame com os inputs, aplica as MESMAS transformações dos
#  Blocos 1 e 2 do script 06 (variáveis derivadas + one-hot esparso) e só então
#  chama predict()
# =============================================================================

library(shiny)
library(Matrix)
library(xgboost)
library(glmnet)
library(ranger)


# =============================================================================
# 1. PARÂMETROS
# =============================================================================

DIR_ARTEFATOS <- "artefatos"   # script 06: "artefatos" (base toda) ou "artefatos_amostra"
CORTE         <- 0.5           # corte de probabilidade do script 06 (alvo ~54/46)
N_CAMPOS      <- 8             # quantas variáveis MAIS IMPORTANTES o formulário pede
VARS_FIXAS    <- c("sexo", "idade_momento_reacao")   # sempre pedidas, além das N mais importantes
VARS_IGNORADAS <- c("notificador")   # nunca viram campo; o modelo as recebe com o valor neutro
MAX_NUMERO    <- 200           # teto das listas numéricas (evita lista enorme se houver outlier)

# Valores que o usuário pode escolher em cada lista (o modelo continua conhecendo todos
# os níveis). Sexo: só Masculino e Feminino, sem "Nao Informado"/"Desconhecido".
# A comparação não diferencia maiúsculas de minúsculas.
SO_ESTES      <- list(sexo = c("Masculino", "Feminino"))

ARQ_MODELOS    <- c(xgboost       = "modelo_xgboost.rds",
                    elasticnet    = "modelo_elasticnet.rds",
                    random_forest = "modelo_random_forest.rds")
ROTULO_MODELOS <- c(xgboost       = "XGBoost",
                    elasticnet    = "Elastic Net",
                    random_forest = "Random Forest")

caminho <- function(arquivo) file.path(DIR_ARTEFATOS, arquivo)


# =============================================================================
# 2. CARGA DOS ARTEFATOS (roda uma vez por processo, compartilhado entre sessões)
# =============================================================================

arquivos <- c(ARQ_MODELOS, "01_treino.rds", "02_teste_esparso.rds", "importancia_variaveis.csv")
faltando <- arquivos[!file.exists(caminho(arquivos))]
if (length(faltando) > 0) {
  stop("Arquivos não encontrados em '", DIR_ARTEFATOS, "/': ", paste(faltando, collapse = ", "))
}

MODELOS <- lapply(ARQ_MODELOS, function(f) readRDS(caminho(f)))

# Nomes (e ordem) das colunas da matriz com que os modelos foram treinados
COLUNAS_MODELO <- colnames(readRDS(caminho("02_teste_esparso.rds"))$X_te)

# Treino em formato data.frame e importância das variáveis: servem só para montar o
# formulário e são descartados ao fim da seção 3.
treino      <- readRDS(caminho("01_treino.rds"))
importancia <- read.csv(caminho("importancia_variaveis.csv"))


# =============================================================================
# 3. VARIÁVEIS DO FORMULÁRIO
# =============================================================================

ROTULOS <- c(
  sexo                        = "Sexo",
  idade_momento_reacao        = "Idade na reação (anos)",
  ind_gestante                = "Gestante",
  ind_lactante                = "Lactante",
  uf                          = "UF",
  tipo_entrada_vigimed        = "Tipo de entrada no VigiMed",
  recebido_de                 = "Recebido de",
  tipo_notificacao            = "Tipo de notificação",
  notificador                 = "Notificador",
  ano_inclusao_sistema        = "Ano de inclusão no sistema",
  reacao_soc                  = "Reação (classe de sistema/órgão MedDRA)",
  qtd_reacoes                 = "Nº de reações notificadas",
  desfecho                    = "Desfecho",
  principios_ativos           = "Princípio(s) ativo(s)",
  atc_nivel1                  = "Grupo anatômico ATC (1º nível)",
  relacao_medicamento_evento  = "Relação medicamento-evento",
  acao_adotada                = "Ação adotada",
  qtd_medicamentos            = "Nº de medicamentos notificados",
  ind_medicamento_no_catalogo = "Medicamento encontrado no catálogo",
  ind_registro_ativo          = "Registro ativo",
  categoria_regulatoria       = "Categoria regulatória",
  classe_terapeutica          = "Classe terapêutica",
  empresa_detentora_registro  = "Empresa detentora do registro"
)

# Tipos de lista. As demais variáveis são categóricas (níveis fixos do treino:
# top 30 mais frequentes + "Outros").
VARS_FLAG      <- c("ind_gestante", "ind_lactante", "ind_medicamento_no_catalogo", "ind_registro_ativo")
VARS_NUMERICAS <- c("idade_momento_reacao", "qtd_reacoes", "qtd_medicamentos")

AJUDAS <- c(idade_momento_reacao = "49 = idade não informada (valor que a base usou para preencher nulos).")

# ---- 3.1 Variáveis que o modelo usa -------------------------------------------
# Variáveis "brutas": as derivadas (faixa_etaria, idade_imputada, log_*) são calculadas
# em prepara_features(). "desfecho" só existe se USAR_DESFECHO = TRUE no script 06.
VARS_MODELO <- setdiff(sub("^log_", "", names(treino)), c("ind_grave", "faixa_etaria", "idade_imputada"))

desconhecidas <- setdiff(VARS_MODELO, names(ROTULOS))
if (length(desconhecidas) > 0) {
  stop("Variáveis do modelo que o app não conhece: ", paste(desconhecidas, collapse = ", "))
}
VARS_MODELO      <- intersect(names(ROTULOS), VARS_MODELO)   # mesma ordem de ROTULOS
VARS_CATEGORICAS <- setdiff(VARS_MODELO, c(VARS_FLAG, VARS_NUMERICAS))

# ---- 3.2 Variáveis do formulário: as fixas + as N mais importantes -----------
# O CSV chama a idade de "idade" (agrupa idade, faixa_etaria e idade_imputada) e as
# contagens de log_qtd_*; aqui os nomes voltam aos do formulário.
importancia$campo <- sub("^log_", "", importancia$variavel)
importancia$campo[importancia$campo %in% c("idade", "faixa_etaria", "idade_imputada")] <- "idade_momento_reacao"

ranking       <- aggregate(queda_auc ~ campo, data = importancia, FUN = mean)   # média dos 3 modelos
ranking       <- ranking[order(-ranking$queda_auc), ]
# Variáveis ignoradas não entram no ranking: a vaga vai para a próxima mais importante.
elegiveis        <- setdiff(VARS_MODELO, VARS_IGNORADAS)
mais_importantes <- head(ranking$campo[ranking$campo %in% elegiveis], N_CAMPOS)

# Fixas primeiro; se uma delas também está entre as mais importantes, não repete.
VARS_EXIBIDAS <- union(intersect(VARS_FIXAS, elegiveis), mais_importantes)

# ---- 3.3 Opções de cada lista (todas as variáveis do modelo) -------------------
# Numéricas: valores inteiros observados no treino. (O treino guarda log1p() das
# contagens, por isso o expm1().)
numericas_treino <- list(
  idade_momento_reacao = treino$idade_momento_reacao,
  qtd_reacoes          = expm1(treino$log_qtd_reacoes),
  qtd_medicamentos     = expm1(treino$log_qtd_medicamentos)
)

OPCOES <- lapply(setNames(nm = VARS_MODELO), function(v) {
  if (v %in% VARS_FLAG) {
    c("Não" = "0", "Sim" = "1")
  } else if (v %in% VARS_NUMERICAS) {
    n <- sort(unique(round(numericas_treino[[v]])))
    as.character(n[n >= 0 & n <= MAX_NUMERO])
  } else {
    # Níveis do treino; descarta os que não viraram coluna do modelo (sem linhas no treino)
    n <- levels(treino[[v]])
    n[make.names(paste0(v, n)) %in% COLUNAS_MODELO]
  }
})

# ---- 3.4 Valor neutro das variáveis que o formulário NÃO pede -----------------
# Segue a convenção da Gold para "desconhecido": flag = 0, categórica = "Nao Informado"
# (se não existir, a categoria mais frequente) e numérica = mediana.
valor_neutro <- function(v) {
  if (v %in% VARS_FLAG) {
    "0"
  } else if (v %in% VARS_NUMERICAS) {
    as.character(round(median(numericas_treino[[v]])))
  } else if ("Nao Informado" %in% OPCOES[[v]]) {
    "Nao Informado"
  } else {
    OPCOES[[v]][1]
  }
}
NEUTRO <- lapply(setNames(nm = VARS_MODELO), valor_neutro)

# ---- 3.5 Valores que o usuário pode escolher ----------------------------------
# O modelo conhece todos os níveis (OPCOES), mas algumas listas oferecem só alguns
# (SO ESTES). As demais oferecem todos os valores da base. Não há filtro entre as
# listas: qualquer combinação de valores pode ser montada.
OFERTA <- lapply(setNames(nm = VARS_EXIBIDAS), function(v) {
  if (!v %in% names(SO_ESTES)) return(OPCOES[[v]])

  oferecidos <- OPCOES[[v]][tolower(OPCOES[[v]]) %in% tolower(SO_ESTES[[v]])]
  if (length(oferecidos) < length(SO_ESTES[[v]])) {
    stop("Não achei todas as categorias de '", v, "' (", paste(SO_ESTES[[v]], collapse = ", "),
         "). Níveis no treino: ", paste(OPCOES[[v]], collapse = ", "), ". Ajuste SO_ESTES.")
  }
  oferecidos
})

rm(treino, importancia, ranking, elegiveis, mais_importantes, numericas_treino,
   desconhecidas, valor_neutro, arquivos, faltando); invisible(gc())


# =============================================================================
# 4. PIPELINE: inputs -> features -> matriz esparsa -> predict()
# =============================================================================

# Mesmas variáveis derivadas do Bloco 1.2 do script 06.
# `d`: data.frame de 1 linha com TODAS as variáveis do modelo (em texto, como vêm das listas).
prepara_features <- function(d) {
  for (v in VARS_CATEGORICAS) d[[v]] <- factor(d[[v]], levels = OPCOES[[v]])
  for (v in c(VARS_FLAG, VARS_NUMERICAS)) d[[v]] <- as.numeric(d[[v]])

  d$idade_imputada       <- as.integer(d$idade_momento_reacao == 49)   # 49 = mediana da Gold
  d$faixa_etaria         <- cut(d$idade_momento_reacao, right = FALSE,
                                breaks = c(-Inf, 2, 12, 18, 40, 60, 80, Inf),
                                labels = c("0-1", "2-11", "12-17", "18-39", "40-59", "60-79", "80+"))
  d$log_qtd_reacoes      <- log1p(d$qtd_reacoes)
  d$log_qtd_medicamentos <- log1p(d$qtd_medicamentos)

  d[setdiff(names(d), c("qtd_reacoes", "qtd_medicamentos"))]
}

# Mesma codificação do Bloco 2 do script 06: one-hot completo em matriz esparsa e
# nomes seguros. No fim, deixa só as colunas do treino, na mesma ordem.
codifica <- function(d) {
  fatores <- names(d)[sapply(d, is.factor)]
  X <- sparse.model.matrix(~ . - 1, data = d,
                           contrasts.arg = lapply(d[fatores], contrasts, contrasts = FALSE),
                           row.names = FALSE)
  colnames(X) <- make.names(colnames(X), unique = TRUE)

  faltam <- setdiff(COLUNAS_MODELO, colnames(X))
  if (length(faltam) > 0) {
    stop("Colunas do modelo ausentes na entrada: ", paste(head(faltam, 5), collapse = ", "))
  }
  X[, COLUNAS_MODELO, drop = FALSE]
}

# Probabilidade de Grave de cada modelo (mesmas chamadas do Bloco 4 do script 06)
PREVER <- list(
  xgboost       = function(m, X) predict(m, xgboost::xgb.DMatrix(X)),
  elasticnet    = function(m, X) as.numeric(predict(m, newx = X, s = "lambda.min", type = "response")),
  random_forest = function(m, X) predict(m, data = X, num.threads = 1)$predictions
)

classifica <- function(entrada, nome_modelo) {
  X    <- codifica(prepara_features(entrada))
  prob <- PREVER[[nome_modelo]](MODELOS[[nome_modelo]], X)
  list(modelo = nome_modelo, prob = prob, grave = prob >= CORTE)
}


# =============================================================================
# 5. UI
# =============================================================================

CSS <- "
  body       { background: #f3f4f6; }
  /* Coluna única com largura de celular, centralizada (no celular ocupa a tela toda) */
  .container-fluid { max-width: 480px; margin: 0 auto; padding: 12px; }
  h2         { font-size: 20px; font-weight: 700; margin: 4px 0 12px; }
  .cartao    { background: #fff; border: 1px solid #dde1e6; border-radius: 10px; padding: 14px 14px 6px; margin-bottom: 12px; }
  .cartao h4 { margin-top: 0; font-weight: 600; }

  /* Alvos de toque confortáveis; fonte de 16px evita o zoom automático do iOS ao focar um campo */
  .form-control, .selectize-input, .selectize-dropdown { font-size: 16px; }
  .selectize-input { min-height: 44px; padding: 10px 12px; }
  .botoes    { display: flex; gap: 8px; margin: 4px 0 12px; }
  .botoes .btn { flex: 1; min-height: 46px; font-size: 16px; }

  .resultado { padding: 12px 14px; border-radius: 6px; background: #f7f8fa; border-left: 6px solid; }
  .resultado.grave     { border-color: #eb6834; }
  .resultado.nao-grave { border-color: #2a78d6; }
  .resultado .classe   { font-size: 26px; font-weight: 700; }
  .resultado.grave .classe     { color: #b8430f; }
  .resultado.nao-grave .classe { color: #1f5fae; }
  .resultado .detalhe  { margin-top: 6px; font-size: 12px; color: #5f6b76; }
  .barra     { height: 8px; margin: 8px 0; background: #e3e6ea; border-radius: 4px; overflow: hidden; }
  .barra div { height: 100%; background: #eb6834; }
"

# A 1ª opção ("" = vazio) aparece como placeholder e é o que o botão "Limpar" seleciona.
# O formulário abre vazio: nenhuma variável entra no modelo sem ser escolhida.
campo_ui <- function(v) {
  tagList(
    selectInput(v, ROTULOS[[v]], choices = c("Selecione..." = "", OFERTA[[v]]), width = "100%"),
    if (v %in% names(AJUDAS)) helpText(AJUDAS[[v]])
  )
}

ui <- fluidPage(
  tags$head(
    tags$meta(name = "viewport", content = "width=device-width, initial-scale=1"),
    tags$style(HTML(CSS))),
  titlePanel("VigiMed | Gravidade da notificação"),

  # Sexo e idade primeiro; depois as variáveis mais importantes, da maior para a menor
  div(class = "cartao",
      h4("Variáveis da notificação"),
      helpText(paste("Sexo e idade são sempre pedidos; as demais são as", N_CAMPOS, "mais importantes.",
                     "As listas são independentes: qualquer combinação de valores pode ser montada.")),
      lapply(VARS_EXIBIDAS, campo_ui)),

  # Modelo, botões e resultado: logo abaixo dos campos, então no celular o resultado
  # aparece onde o dedo acabou de tocar
  div(class = "cartao",
      h4("Classificação"),
      selectInput("modelo", "Modelo", width = "100%",
                  choices = setNames(names(ROTULO_MODELOS), ROTULO_MODELOS)),
      div(class = "botoes",
          actionButton("classificar", "Classificar", class = "btn-primary"),
          actionButton("limpar", "Limpar campos", class = "btn-default")),
      uiOutput("resultado"),
      helpText(paste('"Outros" reúne as categorias raras (fora das 30 mais frequentes do treino).',
                     'As variáveis fora do formulário entram como "Nao Informado"',
                     "(mediana nas numéricas, Não nas de Sim/Não).")))
)


# =============================================================================
# 6. SERVER
# =============================================================================

server <- function(input, output, session) {

  # ---- Classificar -------------------------------------------------------------
  # TRUE depois de "Classificar"; volta a FALSE em "Limpar" (some com o resultado antigo)
  mostrar <- reactiveVal(FALSE)
  observeEvent(input$classificar, mostrar(TRUE))

  # Só roda ao clicar em "Classificar"
  resultado <- eventReactive(input$classificar, {
    escolhidos <- lapply(setNames(nm = VARS_EXIBIDAS), function(v) input[[v]])
    ok         <- vapply(escolhidos, function(x) length(x) == 1 && !is.na(x) && nzchar(x), logical(1))
    validate(need(all(ok), sprintf("Preencha todos os campos (faltam %d).", sum(!ok))))

    # inputs + valor neutro das demais variáveis -> data.frame de 1 linha
    valores <- NEUTRO
    valores[VARS_EXIBIDAS] <- escolhidos
    entrada <- as.data.frame(valores, stringsAsFactors = FALSE)

    classifica(entrada, input$modelo)
  })

  # ---- Limpar campos -----------------------------------------------------------
  # Esvazia as variáveis e some com o resultado. O modelo escolhido não é variável
  # preditora, então continua.
  observeEvent(input$limpar, {
    for (v in VARS_EXIBIDAS) updateSelectInput(session, v, selected = "")
    mostrar(FALSE)
  })

  output$resultado <- renderUI({
    req(mostrar())
    r <- resultado()
    div(class = paste("resultado", if (r$grave) "grave" else "nao-grave"),
        div(class = "classe", if (r$grave) "Grave" else "Não grave"),
        div(class = "barra", div(style = sprintf("width: %.0f%%;", 100 * r$prob))),
        div("Probabilidade de grave: ",
            strong(paste0(format(round(100 * r$prob, 1), nsmall = 1, decimal.mark = ","), "%"))),
        div(class = "detalhe",
            paste0(ROTULO_MODELOS[[r$modelo]], " | corte = ", format(CORTE, decimal.mark = ","))))
  })
}


# =============================================================================
# 7. APP
# =============================================================================

shinyApp(ui, server)
