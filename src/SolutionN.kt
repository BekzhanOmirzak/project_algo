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

    val root = TreeMapNode("")
    repeat(readInt()) {
        val path = readString().split('/')
        var cur = root
        for (p in path) {
            if (cur.children[p] == null) {
                cur.children[p] = TreeMapNode(p)
            }
            cur = cur.children[p]!!
        }
    }

    val paths = mutableListOf<String>()
    dfs(0, root, paths)

    for (p in paths)
        println(p)
}

fun dfs(level: Int, at: TreeMapNode, paths: MutableList<String>) {
    val keys = at.children.keys.sorted()
    for (k in keys) {
        val value = " ".repeat(level) + k
        paths.add(value)
        dfs(level + 2, at.children[k]!!, paths)
    }
}

class TreeMapNode(val value: String) {
    val children: HashMap<String, TreeMapNode> = hashMapOf()
}











