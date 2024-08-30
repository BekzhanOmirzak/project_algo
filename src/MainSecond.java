import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;

public class MainSecond {

    private static String readString() {
        Scanner scanner = new Scanner(System.in);
        return scanner.nextLine();
    }

    public static void main(String[] args) {
        solveProblemA(readString());
    }

    public static void solveProblemA(String str) {
        Set<Integer> set = new HashSet<>();
        for (char c : str.toCharArray()) {
            if (Character.isLowerCase(c)) {
                set.add(1);
            } else if (Character.isDigit(c)) {
                set.add(2);
            } else {
                set.add(3);
            }
        }
        if (set.size() == 3 && str.length() >= 8) {
            System.out.println("YES");
        } else {
            System.out.println("NO");
        }
    }
}
