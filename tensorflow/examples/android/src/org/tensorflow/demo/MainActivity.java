package org.tensorflow.demo;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.app.Activity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import org.tensorflow.demo.R;



public class MainActivity extends Activity {
    public static final String EXTRA_MESSAGE = "org.tensorflow.demo.MESSAGE";
    private Button to_detection;
    private EditText number_collector;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        to_detection = (Button) findViewById(R.id.to_detection);
        number_collector = (EditText) findViewById(R.id.number_collector);

        to_detection.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        Intent intent = new Intent(view.getContext(), DetectorActivity.class);
                        String user_number = number_collector.getText().toString();
                        intent.putExtra(EXTRA_MESSAGE, user_number);
                        startActivity(intent);
                    }
                });

    }


//    public void sendNumber(View view) {
//        Intent intent = new Intent(this, DetectorActivity.class);
//        number_collector = (EditText) findViewById(R.id.number_collector);
//        String user_number = number_collector.getText().toString();
//        intent.putExtra(EXTRA_MESSAGE, user_number);
//        startActivity(intent);
//    }
}
