package com.github.sshaddicts.lucrecium.datasets;

import java.util.List;

public class DataSet<T> {

    private List<T> data;

    public T get(int i){
        return data.get(i);
    }

    public void add(T object){
        data.add(object);
    }

    public void fillDataSet(String directory){
    }
}
