package utils.aho

import java.util.LinkedList


fun main() {
    val ac = AhoCorasick()

    // Add patterns
//    ac.addPattern("abc")
//    ac.addPattern("bc")
//    ac.addPattern("c")
//    ac.addPattern("dcd")

    ac.buildFailureLink()

    val text = "abcdcd"
    val results = ac.search(text)

}

class AhoCorasick {

    val root = TrieNode()

    data class TrieNode(val cost: Int = -1) {
        val children: MutableMap<Char, TrieNode> = mutableMapOf()
        var failureLink: TrieNode? = null
        var outputs: MutableList<Pair<String, Int>> = mutableListOf()
    }

    fun addPattern(pattern: String, cost: Int) {
        var currentNode = root
        for (char in pattern) {
            currentNode = currentNode.children.getOrPut(char) { TrieNode(cost) }
        }
        currentNode.outputs.add(Pair(pattern, cost))
    }

    fun buildFailureLink() {
        val queue = LinkedList<TrieNode>()
        root.children.forEach {
            it.value.failureLink = root
            queue.add(it.value)
        }

        while (queue.isNotEmpty()) {
            val node = queue.poll()

            node.children.forEach { (char, childNode) ->

                var failureNode = node.failureLink
                while (failureNode != null && !failureNode.children.containsKey(char)) {
                    failureNode = failureNode.failureLink
                }

                childNode.failureLink = failureNode?.children?.get(char) ?: root
                childNode.outputs.addAll(childNode.failureLink?.outputs ?: emptyList())
                queue.add(childNode)
            }
        }
    }

    fun search(text: String): List<StringPos> {
        var cur: TrieNode? = root
        val ans = mutableListOf<StringPos>()
        for (i in 0 until text.length) {
            val ch = text[i]

            while (cur != null && cur.children[ch] == null) {
                cur = cur.failureLink
            }

            if (cur == null) {
                cur = root
                continue
            }

            cur = cur.children[ch]!!

            for ((out, cost) in cur.outputs) {
                ans.add(StringPos(i - out.length + 1, out, cost))
            }
        }

        return ans
    }

    data class StringPos(val start: Int, val str: String, val cost: Int)

}

























