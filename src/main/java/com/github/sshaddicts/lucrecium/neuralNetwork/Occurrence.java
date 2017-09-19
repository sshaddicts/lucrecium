package com.github.sshaddicts.lucrecium.neuralNetwork;

public class Occurrence {
    private String name;
    private Double value;

    public Occurrence(String name, Double value) {
        this.name = name;
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public Double getValue() {
        return value;
    }
}
