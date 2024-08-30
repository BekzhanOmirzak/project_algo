import java.util.*
import kotlin.collections.HashSet

private val mod = Math.pow(10.0, 9.0).toInt() + 5
private val dirs = arrayOf(
    intArrayOf(0, 1), intArrayOf(0, -1), intArrayOf(1, 0), intArrayOf(-1, 0)
)

private fun readInt() = readString().toInt()
private fun readString() = readLine().toString()
private fun readDouble() = readString().toDouble()
private fun readListString() = readString().split(" ")
private fun readListInt() = readString().split(" ").map { it.toInt() }
private fun readListLong() = readString().split(" ").map { it.toLong() }

private fun convertStrToList(str: String) = str.split(" ").map { it.toInt() }

fun main() {

    val hodov = readInt()
    val first = hashSetOf<String>()
    val second = hashSetOf<String>()

    var isFirst = false
    var isSecond = false
    var continued = false

    for (i in 1..hodov) {
        val (x, y) = readListInt()
        val hash = "${x}_${y}"
        if (i % 2 == 1) {
            first.add(hash)
            if (findBoard(first)) {
                isFirst = true
            }
        } else {
            second.add(hash)
            if (findBoard(second)) {
                isSecond = true
            }
        }

        if (i<hodov && (isFirst || isSecond)) {
            continued = true
            break
        }
    }

    if (continued) {
        println("Inattention")
    } else if (isFirst) {
        println("First")
    } else if (isSecond) {
        println("Second")
    } else {
        println("Draw")
    }

}

private fun findBoard(set: HashSet<String>): Boolean {
    var hasFound = false
    for (s in set) {
        val (x, y) = s.split("_").map { it.toInt() }
        val result = mutableSetOf<Int>()
        for (i in 1..4) {
            val right = "${x}_${y + i}"
            val down = "${x + i}_${y}"
            val diagonal = "${x + i}_${y + i}"
            if (!set.contains(right))
                result.add(1)
            if (!set.contains(down))
                result.add(2)
            if (!set.contains(diagonal))
                result.add(3)
            if (result.size == 3)
                break
        }
        if (result.size < 3) {
            hasFound = true
            break
        }
    }
    return hasFound
}


































