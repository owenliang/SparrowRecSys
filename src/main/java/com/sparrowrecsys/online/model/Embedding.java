package com.sparrowrecsys.online.model;

import java.util.ArrayList;

/**
 * Embedding Class, contains embedding vector and related calculation
 */
public class Embedding {
    //embedding vector
    ArrayList<Float> embVector;

    public Embedding(){
        this.embVector = new ArrayList<>();
    }

    public Embedding(ArrayList<Float> embVector){
        this.embVector = embVector;
    }

    public void addDim(Float element){
        this.embVector.add(element);
    }

    public ArrayList<Float> getEmbVector() {
        return embVector;
    }

    public void setEmbVector(ArrayList<Float> embVector) {
        this.embVector = embVector;
    }

    //calculate cosine similarity between two embeddings
    // 求2个向量的cosine距离
    public double calculateSimilarity(Embedding otherEmb){
        if (null == embVector || null == otherEmb || null == otherEmb.getEmbVector()
                || embVector.size() != otherEmb.getEmbVector().size()){
            return -1;
        }
        double dotProduct = 0;
        double denominator1 = 0;
        double denominator2 = 0;
        for (int i = 0; i < embVector.size(); i++){
            dotProduct += embVector.get(i) * otherEmb.getEmbVector().get(i);    // 向量内积
            denominator1 += embVector.get(i) * embVector.get(i);    // 左emb向量元素级平方和
            denominator2 += otherEmb.getEmbVector().get(i) * otherEmb.getEmbVector().get(i);    // 右emb向量元素级平方和
        }
        return dotProduct / (Math.sqrt(denominator1) * Math.sqrt(denominator2));    // 内积 / (左平方和*右平方和)
    }
}
