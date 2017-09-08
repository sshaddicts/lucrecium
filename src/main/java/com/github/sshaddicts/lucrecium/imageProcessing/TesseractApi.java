package com.github.sshaddicts.lucrecium.imageProcessing;

import com.github.sshaddicts.lucrecium.util.FileInteractions;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.lept;
import org.bytedeco.javacpp.tesseract;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import javax.xml.bind.DatatypeConverter;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.javacpp.lept.pixDestroy;

public class TesseractApi {

    private String url;
    private int port;
    private final String moduleName = "getText";
    private Logger log = LoggerFactory.getLogger(this.getClass());


    public TesseractApi(String url, int port) {
        if (url == null || Objects.equals(url, "")) {
            throw new IllegalArgumentException("url cannot be null or empty");
        }
        if (port == 0) {
            log.info("switching to default port 8080...");
            port = 8080;
        }

        this.url = url;
        this.port = port;
    }

    public String sendQuery(Mat image) throws IOException {
        String encodedString = DatatypeConverter.printBase64Binary(ImageProcessor.toByteArray(image));

        URL url = formUri(encodedString);

        URLConnection connection = url.openConnection();

        BufferedReader in = new BufferedReader(new InputStreamReader(
                connection.getInputStream()));
        String inputLine;
        StringBuilder sb = new StringBuilder();
        while ((inputLine = in.readLine()) != null)
            sb.append(inputLine).append(" ");
        in.close();

        return sb.toString();
    }

    private URL formUri(String value) throws MalformedURLException {
        String uriBuilder =
                "http://" +
                        url +
                        ":" +
                        port +
                        "/" +
                        moduleName +
                        "/" +
                        value;

        return new URL(uriBuilder);
    }

    public String getTextFromMats(List<Mat> mats) {
        BytePointer outText = new BytePointer();
        lept.PIX image = new lept.PIX();
        tesseract.TessBaseAPI api = new tesseract.TessBaseAPI();

        api.SetPageSegMode(7);

        StringBuilder total = new StringBuilder();

        for (int i = 0; i < mats.size(); i++) {
            Mat mat = mats.get(i);

            ByteArrayOutputStream os = new ByteArrayOutputStream();

            try {
                ImageIO.write(ImageProcessor.toBufferedImage(mat), "bmp", os);
            } catch (IOException e) {
                log.error(e.getMessage(), e);
            }

            api.Init("/usr/share/tesseract-ocr/", "ukr");

            byte[] imageBuffer = os.toByteArray();

            image = lept.pixReadMemBmp(imageBuffer, imageBuffer.length);

            api.SetImage(image);
            outText = api.GetUTF8Text();
            total.append(outText.getString());

            FileInteractions.saveMatWithName(mat, "output", outText.getString());
        }

        api.End();
        outText.deallocate();
        pixDestroy(image);

        return total.toString();
    }

}
