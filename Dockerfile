FROM sshaddicts/neuralswarm
ADD netFileModLetters.lucrecium /netFile
ADD libopencv_java320.so /usr/lib/libopencv_java320.so
ADD target/lucrecium-core-0.2.1-jar-with-dependencies.jar /lek.jar
ADD testCase/best.jpg /test.jpg

ENTRYPOINT ["java", "-jar", "/lek.jar"]
