#0. Przekształcenie plików txt na csv

# 1. Wczytaj potrzebne pakiety
#install.packages("tidyverse")
library(tidyverse)



#Wczytaj listę plików .txt z folderu
sciezka <- "opinie"  # nazwa folderu z plikami
pliki <- list.files(path = sciezka, pattern = "\\.txt$", full.names = TRUE)

#Wczytaj wszystkie pliki i utwórz ramkę danych
opinie_df <- data.frame(
  ID = 1:length(pliki),
  Review_Text = sapply(pliki, function(f) paste(readLines(f, encoding = "UTF-8"), collapse = " ")),
  stringsAsFactors = FALSE
)

#Zapisz do pliku CSV
write.csv(opinie_df, file = "opinie_hoteli.csv", row.names = FALSE, fileEncoding = "UTF-8")




#1. Potrzebne pakiety
required_packages <- c(
  "tm", "SnowballC", "wordcloud", "tidyverse", "tidytext",
  "cluster", "factoextra", "RColorBrewer", "ggrepel", "DT", "rmarkdown", "knitr"
)

lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
})

#2. Wyczyść i przygotuj korpus

# Wczytaj dane (zmień nazwę pliku i kolumny jeśli trzeba)
data <- read.csv("opinie_hoteli.csv", stringsAsFactors = FALSE, encoding = "UTF-8")
corpus <- VCorpus(VectorSource(data$Review_Text))  

# Konwersja do UTF-8
corpus <- tm_map(corpus, content_transformer(function(x) iconv(x, to = "UTF-8", sub = "byte")))

# Funkcja do zamiany znaków na spacje
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))

# Usuwanie zbędnych znaków
corpus <- tm_map(corpus, toSpace, "@")
corpus <- tm_map(corpus, toSpace, "@\\w+")
corpus <- tm_map(corpus, toSpace, "\\|")
corpus <- tm_map(corpus, toSpace, "[ \t]{2,}")
corpus <- tm_map(corpus, toSpace, "(s?)(f|ht)tp(s?)://\\S+\\b")
corpus <- tm_map(corpus, toSpace, "http\\w*")
corpus <- tm_map(corpus, toSpace, "/")
corpus <- tm_map(corpus, toSpace, "(RT|via)((?:\\b\\W*@\\w+)+)")
corpus <- tm_map(corpus, toSpace, "www")
corpus <- tm_map(corpus, toSpace, "~")
corpus <- tm_map(corpus, toSpace, "â€“")

# Standaryzacja tekstu
corpus <- tm_map(corpus, content_transformer(tolower))      # małe litery
corpus <- tm_map(corpus, removeNumbers)                     # usuń liczby
corpus <- tm_map(corpus, removeWords, stopwords("english")) # usuń stopwords
corpus <- tm_map(corpus, removePunctuation)                 # usuń interpunkcję
corpus <- tm_map(corpus, stripWhitespace)                   # zbędne spacje

#Usuń często pojawiające się słowa, które nic nie wnoszą
corpus <- tm_map(corpus, removeWords, c("hotel", "stay"))   # <-- dostosuj, jeśli trzeba

#3. Utwórz Document-Term Matrix z TF-IDF

dtm_tfidf <- DocumentTermMatrix(corpus, 
                                control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))
dtm_tfidf_m <- as.matrix(dtm_tfidf)


#4. Usuń kolumny z zerową wariancją
dtm_tfidf_m <- dtm_tfidf_m[, apply(dtm_tfidf_m, 2, var) != 0]



#5. Dobierz liczbę klastrów
library(factoextra)
fviz_nbclust(dtm_tfidf_m, kmeans, method = "silhouette", k.max = min(6, nrow(dtm_tfidf_m) - 1))


#6. Klastrowanie k-means na TF-IDF
set.seed(123)
k <- 2  # lub inna liczba na podstawie silhouette
kmeans_result <- kmeans(dtm_tfidf_m, centers = k)


#7. Wizualizacja i analiza klastrów
fviz_cluster(list(data = dtm_tfidf_m, cluster = kmeans_result$cluster), 
             geom = "point",
             main = "Klastrowanie opinii TF-IDF")

#8. Przypisanie dokumentów do klastrów
documents_clusters <- data.frame(
  Dokument = 1:nrow(dtm_tfidf_m),  # lub np. data$Author
  Klaster = kmeans_result$cluster
)
print(documents_clusters)


#9. Podsumowanie słów w każdym klastrze
for (i in 1:k) {
  cluster_docs_idx <- which(kmeans_result$cluster == i)
  cluster_docs <- dtm_tfidf_m[cluster_docs_idx, , drop = FALSE]
  word_freq <- sort(colSums(cluster_docs), decreasing = TRUE)
  wordcloud(names(word_freq), freq = word_freq, 
            max.words = 15, colors = brewer.pal(8, "Dark2"))
  title(paste("Chmura słów - Klaster", i))
}




