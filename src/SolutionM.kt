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

    val input = readListString()
    val n = input[0].toInt()
    val dir = input[0]

    val matrix = mutableListOf<List<Long>>()
    repeat(n) {
        matrix.add(readListLong())
    }

    val rotated: Array<Array<Long>>

    if (dir == "R") {
        rotated = rotateToRight(matrix)
    } else {
        rotated = rotateToLeft(matrix)
    }

    solveWithSwap(matrix, rotated)

}

fun solveWithSwap(cur: List<List<Long>>, target: Array<Array<Long>>) {

    val state = cur.map { it.toTypedArray() }.toTypedArray()

    val diff = mutableListOf<List<Int>>()
    for (i in 0 until cur.size) {
        for (j in 0 until cur.size) {
            if (state[i][j] != target[i][j]) {
                diff.add(listOf(i, j))
            }
        }
    }

    val answer = mutableListOf<List<Int>>()

    for (k in 0 until diff.size) {
        val (i, j) = diff[k]
        if (state[i][j] != target[i][j]) {
            val need = target[i][j]
            for (l in k + 1 until diff.size) {
                val (nI, nJ) = diff[l]
                if (state[nI][nJ] == need) {
                    answer.add(listOf(i, j, nI, nJ))
                    val old = state[i][j]
                    state[i][j] = state[nI][nJ]
                    state[nI][nJ] = old
                }
            }
        }
    }

    println(answer.size)

    for (a in answer) {
        println(a.joinToString(" "))
    }

}


fun rotateToLeft(matrix: List<List<Long>>): Array<Array<Long>> {

    val ans = Array(matrix[0].size) { Array(matrix.size) { 0L } }

    for (i in 0 until matrix.size) {
        val curRow = matrix[i]
        for (j in 0 until curRow.size) {
            ans[curRow.size - 1 - j][i] = curRow[j]
        }
    }

    return ans
}

fun rotateToRight(matrix: List<List<Long>>): Array<Array<Long>> {

    val ans = Array(matrix[0].size) { Array(matrix.size) { 0L } }

    for (i in 0 until matrix.size) {
        val curRow = matrix[i]
        for (j in 0 until curRow.size) {
            ans[j][matrix.size - i - 1] = curRow[j]
        }
    }

    return ans
}