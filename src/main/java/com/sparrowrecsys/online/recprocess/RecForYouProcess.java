package com.sparrowrecsys.online.recprocess;

import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.datamanager.User;
import com.sparrowrecsys.online.datamanager.Movie;
import com.sparrowrecsys.online.datamanager.RedisClient;
import com.sparrowrecsys.online.util.Config;
import com.sparrowrecsys.online.util.Utility;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.*;

import static com.sparrowrecsys.online.util.HttpClient.asyncSinglePostRequest;

/**
 * Recommendation process of similar movies
 */

public class RecForYouProcess {

    /**
     * get recommendation movie list
     * @param userId input user id
     * @param size  size of similar items
     * @param model model used for calculating similarity
     * @return  list of similar movies
     */
    public static List<Movie> getRecList(int userId, int size, String model){
        // 用户
        User user = DataManager.getInstance().getUserById(userId);
        if (null == user){
            return new ArrayList<>();
        }

        // 平均分TOP 800的电影候选集
        final int CANDIDATE_SIZE = 800;
        List<Movie> candidates = DataManager.getInstance().getMovies(CANDIDATE_SIZE, "rating");

        //load user emb from redis if data source is redis
        // 开关：从redis实时加载user embedding，默认已经加载到User对象里了
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_REDIS)){
            String userEmbKey = "uEmb:" + userId;
            String userEmb = RedisClient.getInstance().get(userEmbKey);
            if (null != userEmb){
                user.setEmb(Utility.parseEmbStr(userEmb));
            }
        }

        // 开关：从redis实时加载user features，默认是没有从文件加载到的
        if (Config.IS_LOAD_USER_FEATURE_FROM_REDIS){
            String userFeaturesKey = "uf:" + userId;
            Map<String, String> userFeatures = RedisClient.getInstance().hgetAll(userFeaturesKey);
            if (null != userFeatures){
                user.setUserFeatures(userFeatures);
            }
        }

        // 调用模型打分
        List<Movie> rankedList = ranker(user, candidates, model);

        if (rankedList.size() > size){
            return rankedList.subList(0, size);
        }
        return rankedList;
    }

    /**
     * rank candidates
     * @param user    input user
     * @param candidates    movie candidates
     * @param model     model name used for ranking
     * @return  ranked movie list
     */
    public static List<Movie> ranker(User user, List<Movie> candidates, String model){
        HashMap<Movie, Double> candidateScoreMap = new HashMap<>();

        switch (model){
            // 直接计算user和各个movie之间的emb距离
            case "emb":
                for (Movie candidate : candidates){
                    double similarity = calculateEmbSimilarScore(user, candidate);
                    candidateScoreMap.put(candidate, similarity);
                }
                break;
            case "nerualcf":    // tf-serving模型打分
                callNeuralCFTFServing(user, candidates, candidateScoreMap);
                break;
            default:
                //default ranking in candidate set
                for (int i = 0 ; i < candidates.size(); i++){
                    candidateScoreMap.put(candidates.get(i), (double)(candidates.size() - i));
                }
        }

        List<Movie> rankedList = new ArrayList<>();
        candidateScoreMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).forEach(m -> rankedList.add(m.getKey()));
        return rankedList;
    }

    /**
     * function to calculate similarity score based on embedding
     * @param user     input user
     * @param candidate candidate movie
     * @return  similarity score
     */
    // 计算user和movie的embedding余弦距离
    public static double calculateEmbSimilarScore(User user, Movie candidate){
        if (null == user || null == candidate || null == user.getEmb()){
            return -1;
        }
        return user.getEmb().calculateSimilarity(candidate.getEmb());
    }

    /**
     * call TenserFlow serving to get the NeuralCF model inference result
     * @param user              input user
     * @param candidates        candidate movies
     * @param candidateScoreMap save prediction score into the score map
     */
    public static void callNeuralCFTFServing(User user, List<Movie> candidates, HashMap<Movie, Double> candidateScoreMap){
        if (null == user || null == candidates || candidates.size() == 0){
            return;
        }

        JSONArray instances = new JSONArray();
        // 准备特征{userId, movieId}
        for (Movie m : candidates){
            JSONObject instance = new JSONObject();
            instance.put("userId", user.getUserId());
            instance.put("movieId", m.getMovieId());
            instances.put(instance);
        }

        // 准备tf-serving请求
        JSONObject instancesRoot = new JSONObject();
        instancesRoot.put("instances", instances);

        // 调用tf-serving
        //need to confirm the tf serving end point
        String predictionScores = asyncSinglePostRequest("http://localhost:8501/v1/models/recmodel:predict", instancesRoot.toString());
        System.out.println("send user" + user.getUserId() + " request to tf serving.");

        JSONObject predictionsObject = new JSONObject(predictionScores);
        JSONArray scores = predictionsObject.getJSONArray("predictions");
        // 解析打分结果
        for (int i = 0 ; i < candidates.size(); i++){
            // 将每个打分赋给电影
            candidateScoreMap.put(candidates.get(i), scores.getJSONArray(i).getDouble(0));
        }
    }
}
