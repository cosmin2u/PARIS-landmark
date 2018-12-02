package com.example.enkay.cifar;

import android.content.res.AssetManager;
import android.graphics.Matrix;
import android.support.annotation.IntegerRes;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.util.Arrays;


//import TensorFlow libraries
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

// Import image utilities
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;


public class MainActivity extends AppCompatActivity {


    private TensorFlowInferenceInterface inferenceInterface;
    private static final String MODEL_FILE = "file:///android_asset/frozen_model_CIFARdinproiectgpu.pb";
    /*
    Please note that the name of input and output nodes should match that of names we declared in
    our TensorFlow graph
     */

    private static final String INPUT_NODE = "intrare:0"; // our input node
    private static final String OUTPUT_NODE = "iesire"; // our output node

    private static final int[] INPUT_SIZE = {1,32,32,3};

    static {
        System.loadLibrary("tensorflow_inference");
    }

    // helper function to find the indices of the element in an array with maximum value
    public static int argmax (float [] elems)
    {
        int bestIdx = -1;
        float max = -1000;
        for (int i = 0; i < elems.length; i++) {
            float elem = elems[i];
            if (elem > max) {
                max = elem;
                bestIdx = i;
            }
        }
        return bestIdx;
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
        System.out.println("model loaded successfully");
        String imageUri = "drawable://" + R.drawable.models;

        AssetManager assetManager = getAssets();
        final int image_size = 32;
        try {
            final int inputSize = image_size;
            final int destWidth = image_size;
            final int destHeight = image_size;
            /*
            Mean and standard deviation of the Dataset
            Initialise them with corresponding values

            int imageMean;
            float imageStd;
             */
            // Load the image
            InputStream file = assetManager.open("paris_louvre_000236.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(file);
            Bitmap bitmap_scaled = Bitmap.createScaledBitmap(bitmap, destWidth, destHeight, false);

            // Set the bitmap to the image view (UI) - optional
            ImageView image= (ImageView) findViewById(R.id.car);
            image.setImageBitmap(bitmap);


            // Load class names of CIFAR 10 dataset into a string array
            String[] classes = {"defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame", "pantheon", "pompidou", "sacrecoeur", "triomphe"};



            int[] intValues = new int[inputSize * inputSize]; // array to copy values from Bitmap image
            float[] floatValues = new float[inputSize * inputSize * 3]; // float array to store image data

            // note: Both intValues and floatValues are flattened arrays

            //get pixel values from bitmap image and store it in intValues
            bitmap_scaled.getPixels(intValues, 0, bitmap_scaled.getWidth(), 0, 0, bitmap_scaled.getWidth(), bitmap_scaled.getHeight());
            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];
                /*
                preprocess image if required
                floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
                floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
                floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
                */

                // convert from 0-255 range to floating point value
                floatValues[i * 3 + 0] = ((val >> 16) & 0xFF);
                floatValues[i * 3 + 1] = ((val >> 8) & 0xFF);
                floatValues[i * 3 + 2] = (val & 0xFF);
            }

            //  the input size node that we declared earlier will be a parameter to reshape the tensor
            // fill the input node with floatValues array
            Log.i("mydbg", INPUT_NODE);
            Log.i("mydbg", Arrays.toString(INPUT_SIZE));
            Log.i("mydbg", Arrays.toString(floatValues));
            Log.i("mydbg", Arrays.toString(intValues));
            inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, floatValues);

            // make the inference
            inferenceInterface.runInference(new String[] {OUTPUT_NODE});
            // create an array filled zeros with dimension of number of output classes. In our case its 10
            float [] result = new float[11];
            Arrays.fill(result,0.0f);
            // copy the values from output node to the 'result' array
            inferenceInterface.readNodeFloat(OUTPUT_NODE, result);
            // find the class with highest probability
            int class_id=argmax(result);
            TextView textView=(TextView) findViewById(R.id.result);
            // Setting the class name in the UI
            textView.setText(classes[class_id]);

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
