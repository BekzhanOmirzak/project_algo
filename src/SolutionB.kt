import java.util.*

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

    val target = readString().trim()
    val word = readString().trim()

    val start = LinkedList<Char>()
    val end = LinkedList<Char>()

    var i = 0
    while (i < word.length) {
        val cur = word[i]
        if (cur == '<') {
            i++
            val op = StringBuilder()
            while (i < word.length && word[i] != '>') {
                op.append(word[i])
                i++
            }

            var valid = true

            when (op.toString()) {
                "left" -> {
                    if (start.isNotEmpty())
                        end.add(0, start.pollLast())
                }

                "right" -> {
                    if (end.isNotEmpty()) {
                        start.add(end.pollFirst())
                    }
                }

                "delete" -> {
                    if (end.isNotEmpty())
                        end.pollFirst()
                }

                "bspace" -> {
                    if (start.isNotEmpty())
                        start.pollLast()
                }

                else ->
                    valid = false
            }
            if (valid)
                i++
            else {
                start.add('<')
                op.forEach {
                    start.add(it)
                }
            }
        } else {
            start.add(cur)
            i++
        }
    }

    val ansBuilder = StringBuilder()

    for (cur in start)
        if (cur != ' ')
            ansBuilder.append(cur)

    for (cur in end)
        if (cur != ' ')
            ansBuilder.append(cur)

    if (ansBuilder.toString() == target)
        println("YES")
    else
        println("NO")

}


































