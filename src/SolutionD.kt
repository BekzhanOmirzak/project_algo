import java.util.LinkedList

private val mod = Math.pow(10.0, 9.0).toInt() + 5
private val dirs = arrayOf(
    intArrayOf(0, 1), intArrayOf(0, -1), intArrayOf(1, 0), intArrayOf(-1, 0)
)

private fun readInt() = readString().toInt()
private fun readString() = readLine().toString()
private fun readDouble() = readString().toDouble()
private fun readListString() = readString().split(" ")
private fun readListInt() = readString().split(" ").map { it.toInt() }

private fun convertStrToList(str: String) = str.split(" ").map { it.toInt() }

fun main() {


}


fun minimumArrayLength(nums: IntArray): Int {

    println(
        minimumArrayLength(
            intArrayOf(5, 2, 2, 2, 9, 10)
        )
    )

    nums.sort()
    var at = 0
    while (at < nums.size && nums[at] == nums[0]) {
        at++
    }

    return Math.ceil((at + 1) / 2.0).toInt()
}

fun solve(inDegrees: IntArray, adj: Array<MutableList<Int>>): Int {

    val n = readInt()
    val adj = Array(n) { mutableListOf<Int>() }
    val inDegrees = IntArray(n) { 0 }
    repeat(n) {
        val needs = readListInt()
        if (needs[0] != 0) {
            for (i in 1 until needs.size) {
                val from = needs[i]
                adj[from - 1].add(it)
                inDegrees[it]++
            }
        }
    }

    solve(inDegrees, adj)

    var c = 0
    val queue = LinkedList<Int>()
    for (i in 0 until inDegrees.size) {
        if (inDegrees[i] == 0) {
            queue.add(i)
        }
    }

    val ans = mutableListOf<List<Int>>()

    while (queue.isNotEmpty()) {
        val size = queue.size
        c++
        ans.add(queue.toList().map { it + 1 })
        for (i in 0 until size) {
            val poll = queue.poll()
            for (n in adj[poll]) {
                inDegrees[n]--
                if (inDegrees[n] == 0) {
                    queue.add(n)
                }
            }
        }
    }

    println(c)

    for (a in ans) {
        println("${a.size} ${a.joinToString(" ")}")
    }

    return c
}

fun solve(matrix: List<List<Int>>): Int {

    val n = readInt()
    val rows = mutableListOf<List<Int>>()
    repeat(n) {
        val row = readListInt()
        rows.add(row)
    }

    println(solve(rows))

    val prefixRow = Array(matrix.size) { IntArray(matrix[0].size) { 0 } }
    val prefixCol = Array(matrix.size) { IntArray(matrix[0].size) { 0 } }

    for (r in 0 until matrix.size) {
        var row = 0
        for (c in 0 until matrix.size) {
            row += matrix[r][c]
            prefixRow[r][c] = row
        }
    }

    for (c in 0 until matrix.size) {
        var col = 0
        for (r in 0 until matrix.size) {
            col += matrix[r][c]
            prefixCol[r][c] = col
        }
    }

    var c = 0
    for (i in 0 until matrix.size) {
        for (j in 0 until matrix[i].size) {
            val row = prefixRow[i][matrix.size - 1]
            val col = prefixCol[matrix.size - 1][j]
            val diff = Math.abs(row - col)
            if (diff <= matrix[i][j])
                c++
        }
    }

    return c
}