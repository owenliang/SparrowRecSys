package com.sparrowrecsys.online.recprocess;

import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.datamanager.Movie;
import java.util.*;

/**
 * Recommendation process of similar movies
 */

public class SimilarMovieProcess {

    /**
     * get recommendation movie list
     * @param movieId input movie id
     * @param size  size of similar items
     * @param model model used for calculating similarity
     * @return  list of similar movies
     */
    // 根据电影获取相关电影推荐
    public static List<Movie> getRecList(int movieId, int size, String model){
        Movie movie = DataManager.getInstance().getMovieById(movieId); // 当前电影
        if (null == movie){
            return new ArrayList<>();
        }
        // 候选集
        List<Movie> candidates = candidateGenerator(movie);
        
        // 模型排序
        List<Movie> rankedList = ranker(movie, candidates, model);

        // 保留TOP N
        if (rankedList.size() > size){
            return rankedList.subList(0, size);
        }
        return rankedList;
    }

    /**
     * generate candidates for similar movies recommendation
     * @param movie input movie object
     * @return  movie candidates
     */
    public static List<Movie> candidateGenerator(Movie movie){
        HashMap<Integer, Movie> candidateMap = new HashMap<>();
        for (String genre : movie.getGenres()){ // 通过当前电影的分类召回所有电影
            List<Movie> oneCandidates = DataManager.getInstance().getMoviesByGenre(genre, 100, "rating"); // 每个分类按平均分取TOP 100的电影
            for (Movie candidate : oneCandidates){
                candidateMap.put(candidate.getMovieId(), candidate);
            }
        }
        // 不包含自己
        candidateMap.remove(movie.getMovieId());
        // 返回候选集
        return new ArrayList<>(candidateMap.values());
    }

    /**
     * multiple-retrieval candidate generation method
     * @param movie input movie object
     * @return movie candidates
     */
    // 策略多路召回
    public static List<Movie> multipleRetrievalCandidates(Movie movie){
        if (null == movie){
            return null;
        }

        HashSet<String> genres = new HashSet<>(movie.getGenres());

        // 按电影分类，召回各个分类下高评分电影
        HashMap<Integer, Movie> candidateMap = new HashMap<>();
        for (String genre : genres){
            List<Movie> oneCandidates = DataManager.getInstance().getMoviesByGenre(genre, 20, "rating");
            for (Movie candidate : oneCandidates){
                candidateMap.put(candidate.getMovieId(), candidate);
            }
        }

        // 获取评分最高的100部电影
        List<Movie> highRatingCandidates = DataManager.getInstance().getMovies(100, "rating");
        for (Movie candidate : highRatingCandidates){
            candidateMap.put(candidate.getMovieId(), candidate);
        }

        // 获取上映最新的100部电影
        List<Movie> latestCandidates = DataManager.getInstance().getMovies(100, "releaseYear");
        for (Movie candidate : latestCandidates){
            candidateMap.put(candidate.getMovieId(), candidate);
        }

        candidateMap.remove(movie.getMovieId());
        return new ArrayList<>(candidateMap.values());
    }

    /**
     * embedding based candidate generation method
     * @param movie input movie
     * @param size  size of candidate pool
     * @return  movie candidates
     */
    // 基于embedding的召回
    public static List<Movie> retrievalCandidatesByEmbedding(Movie movie, int size){
        if (null == movie || null == movie.getEmb()){
            return null;
        }

        // 按评分最高的10000个电影
        List<Movie> allCandidates = DataManager.getInstance().getMovies(10000, "rating");
        HashMap<Movie,Double> movieScoreMap = new HashMap<>();
        // 计算每个候选电影的emb相似度
        for (Movie candidate : allCandidates){
            double similarity = calculateEmbSimilarScore(movie, candidate);
            movieScoreMap.put(candidate, similarity);
        }

        // 按emb相似度排序
        List<Map.Entry<Movie,Double>> movieScoreList = new ArrayList<>(movieScoreMap.entrySet());
        movieScoreList.sort(Map.Entry.comparingByValue());

        // 返回最高分的size个候选集
        List<Movie> candidates = new ArrayList<>();
        for (Map.Entry<Movie,Double> movieScoreEntry : movieScoreList){
            candidates.add(movieScoreEntry.getKey());
        }

        return candidates.subList(0, Math.min(candidates.size(), size));
    }

    /**
     * rank candidates
     * @param movie    input movie
     * @param candidates    movie candidates
     * @param model     model name used for ranking
     * @return  ranked movie list
     */
    // 基于embedding做电影相关度排序
    // 计算：当前电影 与 候选电影 之间的embedding距离
    public static List<Movie> ranker(Movie movie, List<Movie> candidates, String model){
        HashMap<Movie, Double> candidateScoreMap = new HashMap<>();
        for (Movie candidate : candidates){
            double similarity;
            switch (model){
                case "emb": // 基于emb向量距离的相似度打分
                    similarity = calculateEmbSimilarScore(movie, candidate);
                    break;
                default:    // 基于策略的相似度打分
                    similarity = calculateSimilarScore(movie, candidate);
            }
            candidateScoreMap.put(candidate, similarity);   // 保存电影-->打分
        }
        List<Movie> rankedList = new ArrayList<>();
        // 按打分从大到小排序，加入到结果列表
        candidateScoreMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).forEach(m -> rankedList.add(m.getKey()));
        return rankedList;
    }

    /**
     * function to calculate similarity score
     * @param movie     input movie
     * @param candidate candidate movie
     * @return  similarity score
     */
    // 基于策略的相似度打分
    public static double calculateSimilarScore(Movie movie, Movie candidate){
        int sameGenreCount = 0; // 2个电影的分类交集有几个？
        for (String genre : movie.getGenres()){
            if (candidate.getGenres().contains(genre)){
                sameGenreCount++;
            }
        }
        double genreSimilarity = (double)sameGenreCount / (movie.getGenres().size() + candidate.getGenres().size()) / 2;    // 求分类相似度
        double ratingScore = candidate.getAverageRating() / 5;  // 候选电影的平均分

        double similarityWeight = 0.7;
        double ratingScoreWeight = 0.3;

        // 分类相似度打分*70% + 候选电影平均分*30%
        return genreSimilarity * similarityWeight + ratingScore * ratingScoreWeight;
    }

    /**
     * function to calculate similarity score based on embedding
     * @param movie     input movie
     * @param candidate candidate movie
     * @return  similarity score
     */
    public static double calculateEmbSimilarScore(Movie movie, Movie candidate){
        if (null == movie || null == candidate){
            return -1;
        }
        // 计算2个电影emb向量之间的距离
        return movie.getEmb().calculateSimilarity(candidate.getEmb());
    }
}
