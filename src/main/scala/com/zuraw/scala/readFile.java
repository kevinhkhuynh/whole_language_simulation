package com.zuraw.scala;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.io.File;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.util.ArrayList;

public class readFile {
    public static ArrayList<String[]> read(String file)
    {
        FileInputStream input;
        ArrayList<String[]> tableau = new ArrayList<String[]>();
        try {
            input = new FileInputStream(new File(file));
            CharsetDecoder decoder = Charset.forName("UTF-8").newDecoder();
            decoder.onMalformedInput(CodingErrorAction.IGNORE);
            InputStreamReader reader = new InputStreamReader(input, decoder);
            BufferedReader bufferedReader = new BufferedReader(reader);
            String line = bufferedReader.readLine();
            while (line != null) {
                tableau.add(line.split(","));
                line = bufferedReader.readLine();
            }
            bufferedReader.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return tableau;
    }
}
