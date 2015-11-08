package com.neetomo.weka;

import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Attribute;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * 今回用意した、titanicデータを予測するやつ
 * なお、このデータに特化した処理だったり、Exceptionは全throwしたり
 */
public class Titanic {
    /** classifier */
    private Classifier classifier;

    /**
     * 学習済みモデルを読み込む
     *
     * @param model
     * @throws Exception
     */
    public void loadMode(String model) throws Exception {
        FileInputStream fis = new FileInputStream(model);
        ObjectInputStream ois = new ObjectInputStream(fis);
        this.classifier  = (Classifier) ois.readObject();

        ois.close();
        fis.close();
    }

    /**
     * モデルの保存
     *
     * @param path
     * @throws Exception
     */
    public void save(String path) throws Exception {
        FileOutputStream fos = new FileOutputStream(path);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(classifier);

        oos.close();
        fos.close();
    }

    /**
     * トレーニング
     *
     * @param data 学習データ(arff)
     * @throws Exception
     */
    public void train(String data) throws Exception {
        DataSource source = new DataSource(data);
        Instances instances = source.getDataSet();
        instances.setClassIndex(3);

        Classifier classifier = new Logistic();
        classifier.buildClassifier(instances);

        this.classifier = classifier;
    }

    /**
     * test用arffファイルを読み込んでの予測
     *
     * @param data
     * @throws Exception
     */
    public void predict(String data) throws Exception {
        DataSource source = new DataSource(data);
        Instances instances = source.getDataSet();
        instances.setClassIndex(3);

        for (Instance instance : instances) {
            System.out.print(instance);
            System.out.print(" : ");
            System.out.println(this.classifier.classifyInstance(instance));

        }
    }

    /**
     * データを一つ受け取って予測結果を返す
     *
     * @param pclass
     * @param age
     * @param sex
     * @throws Exception
     */
    public void predictOne(String pclass, int age, String sex) throws Exception {
        Attribute attrPclass = new Attribute("pclass", Arrays.asList("1", "2", "3"), 0);
        Attribute attrAge = new Attribute("age", 1);
        Attribute attrSex = new Attribute("sex", Arrays.asList("male", "female"), 2);
        Attribute attrSurvived = new Attribute("survived", Arrays.asList("0", "1"), 3);

        ArrayList<Attribute> attrs = new ArrayList<Attribute>();
        attrs.add(attrPclass);
        attrs.add(attrAge);
        attrs.add(attrSex);
        attrs.add(attrSurvived);

        Instances instances = new Instances("Logistic Regression", attrs, 0);
        instances.setClassIndex(3);

        Instance instance = new DenseInstance(4);
        instance.setValue(attrPclass, pclass);
        instance.setValue(attrAge, age);
        instance.setValue(attrSex, sex);
        instance.setDataset(instances);

        System.out.print(instance);
        System.out.print(" : ");
        System.out.println(classifier.classifyInstance(instance));
    }

    public static void main(String[] args) throws Exception {
        Titanic titanic = new Titanic();

        //titanic.train("data/titanic.train.arff");

        //titanic.save("model/titanic.model");
        titanic.loadMode("model/titanic.model");

        titanic.predict("data/titanic.test.arff");
        titanic.predictOne("1", 5, "female");
    }
}
