package com.sparrowrecsys.online;

import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.service.*;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.DefaultServlet;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.util.resource.Resource;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URL;

/***
 * Recsys Server, end point of online recommendation service
 */

public class RecSysServer {

    public static void main(String[] args) throws Exception {
        new RecSysServer().run();
    }

    //recsys server port number
    private static final int DEFAULT_PORT = 6010;

    public void run() throws Exception{

        int port = DEFAULT_PORT;
        try {
            port = Integer.parseInt(System.getenv("PORT"));
        } catch (NumberFormatException ignored) {}

        //set ip and port number
        InetSocketAddress inetAddress = new InetSocketAddress("0.0.0.0", port);
        Server server = new Server(inetAddress);

        //get index.html path
        URL webRootLocation = this.getClass().getResource("/webroot/index.html");
        if (webRootLocation == null)
        {
            throw new IllegalStateException("Unable to determine webroot URL location");
        }

        //set index.html as the root page
        URI webRootUri = URI.create(webRootLocation.toURI().toASCIIString().replaceFirst("/index.html$","/"));
        System.out.printf("Web Root URI: %s%n", webRootUri.getPath());

        //load all the data to DataManager
        DataManager.getInstance().loadData(webRootUri.getPath() + "sampledata/movies.csv",
                webRootUri.getPath() + "sampledata/links.csv",webRootUri.getPath() + "sampledata/ratings.csv",
                webRootUri.getPath() + "modeldata/item2vecEmb.csv",
                webRootUri.getPath() + "modeldata/userEmb.csv",
                "i2vEmb", "uEmb");

        //create server context
        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath("/");
        context.setBaseResource(Resource.newResource(webRootUri));
        context.setWelcomeFiles(new String[] { "index.html" });
        context.getMimeTypes().addMimeMapping("txt","text/plain;charset=utf-8");

        //bind services with different servlets
        context.addServlet(DefaultServlet.class,"/");

        // 获取电影详情，根据ID直接返回movie对象
        context.addServlet(new ServletHolder(new MovieService()), "/getmovie");
        // 电影详情页，推荐相关电影（1，电影之间的策略相似度打分；2，电影之间的emb余弦距离打分；）
        context.addServlet(new ServletHolder(new SimilarMovieService()), "/getsimilarmovie");
        
        // 获取用户信息
        context.addServlet(new ServletHolder(new UserService()), "/getuser");
        // 猜你喜欢（基于model的个性化推荐）
        context.addServlet(new ServletHolder(new RecForYouService()), "/getrecforyou");

        // 首页推荐列表，默认是按电影平均得分排序的，没看到模型
        context.addServlet(new ServletHolder(new RecommendationService()), "/getrecommendation");  

        //set url handler
        server.setHandler(context);
        System.out.println("RecSys Server has started.");

        //start Server
        server.start();
        server.join();
    }
}
