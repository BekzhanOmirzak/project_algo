package test;

import java.util.HashMap;

public class Test {


    public static void main(String[] args) {

        int case1=2;

        switch (case1) {
            case 1:
                System.out.println("1");
            case 2:
                System.out.println("2");
            case 3:
                System.out.println("3");
            case 4:
                System.out.println("4");
            default:
                System.out.println("default");
        }

    }

    static void changeNum(Integer c) {
        for (int i = 0; i < 10; i++) {
            c++;
        }
    }

    static void change(String a) {
        a = "Changed in a function body";
    }

    static void changeObjectLoc(MyClass obj) {
        obj.name = "I am a new name";
        System.out.println("Change Location of object : " + obj);
    }

}

class MyClass {
    String name;

    @Override
    public String toString() {
        return name;
    }
}

class MyKey {

    @Override
    public int hashCode() {
        System.out.println("Checking hash code " + this);
        return 1;
    }

    @Override
    public boolean equals(Object obj) {
        System.out.println("Checking equals " + this);
        return false;
    }

    @Override
    public String toString() {
        return "Hello";
    }
}
