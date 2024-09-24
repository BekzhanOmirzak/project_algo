import java.util.*
import kotlin.collections.HashMap

private val mod = Math.pow(10.0, 9.0).toInt() + 5
private val dirs = arrayOf(
    intArrayOf(-2, -1),
    intArrayOf(-1, -2),
    intArrayOf(1, -2),
    intArrayOf(2, -1),
    intArrayOf(2, 1),
    intArrayOf(1, 2),
    intArrayOf(-1, 2),
    intArrayOf(-2, 1)
)

private fun readInt() = readString().toInt()
private fun readString() = readLine().toString()
private fun readDouble() = readString().toDouble()
private fun readListString() = readString().split(" ")
private fun readListInt() = readString().split(" ").map { it.toInt() }
private fun readListLong() = readString().split(" ").map { it.toLong() }

private fun convertStrToList(str: String) = str.split(" ").map { it.toInt() }

fun main() {

    println(
        minNumberOfSeconds(
            10,
            intArrayOf(3, 2, 2, 4)
        )
    )

}

fun minNumberOfSeconds(mountainHeight: Int, workerTimes: IntArray): Long {

    return 0L
}

fun isEnough(curSecond: Int, workerTimes: IntArray): Int {
    var total = 0
    for (time in workerTimes) {
        var sec = time
        
    }

    return 0
}

fun maxEnvelopes(envelopes: Array<IntArray>): Int {
    println(
        maxEnvelopes(
            arrayOf(
                intArrayOf(5, 4),
                intArrayOf(6, 4),
                intArrayOf(6, 7),
                intArrayOf(2, 3)
            )
        )
    )

    println(
        maxPathLength(
            arrayOf(
                intArrayOf(3, 1),
                intArrayOf(4, 1),
                intArrayOf(0, 0),
                intArrayOf(5, 3),
                intArrayOf(2, 2)
            ),
            1
        )
    )



    envelopes.sortWith { o1, o2 ->
        if (o1[0] == o2[0])
            o1[1] - o2[1]
        else
            o1[0] - o2[0]
    }

    val list = mutableListOf<IntArray>()
    list.add(envelopes[0])

    for (i in 1 until envelopes.size) {
        val (x, y) = envelopes[i]
        val last = list[list.size - 1]
        if (last[0] < x && last[1] < y) {
            list.add(envelopes[i])
        } else {
            val leftMostIndex = leftMostIndex(list, envelopes[i])
            val leftMost = list[leftMostIndex]
            if (leftMost[1] < y) {
                list[leftMostIndex] = envelopes[i]
            }
        }
    }

    return list.size
}

fun leftMostIndex(list: MutableList<IntArray>, last: IntArray): Int {
    var l = 0
    var r = list.size - 1
    var ans = list.size - 1
    while (l <= r) {
        val mid = (l + r) / 2
        val cur = list[mid]
        if (cur[0] >= last[0] && cur[1] >= last[1]) {
            ans = Math.min(ans, mid)
            r = mid - 1
        } else {
            l = mid + 1
        }
    }

    return ans
}

fun bsLeftMostIndex(list: MutableList<IntArray>): Int {

    var l = 0
    var r = list.size - 1


    return 0
}

fun maxPathLength(coordinates: Array<IntArray>, k: Int): Int {

    val initial = coordinates[k]
    coordinates.sortWith { o1, o2 ->
        if (o1[0] == o2[0]) {
            o2[1] - o1[1]
        }
        o1[0] - o2[0]
    }

    var pos = 0
    for (i in 0 until coordinates.size) {
        val cur = coordinates[i]
        if (initial.contentEquals(cur)) {
            pos = i
            break
        }
    }

    val left = leftLIS(coordinates, pos)
    val right = rightLIS(coordinates, pos)
    return left + right + 1
}

fun leftLIS(coordinates: Array<IntArray>, pos: Int): Int {
    var c = 0
    var prev = coordinates[pos]
    for (i in pos downTo 0) {
        val cur = coordinates[i]
        if (cur[0] < prev[0] && cur[1] < prev[1]) {
            prev = cur
            c++
        }
    }

    return c
}

fun rightLIS(coordinates: Array<IntArray>, pos: Int): Int {
    var c = 0
    var prev = coordinates[pos]
    for (i in pos until coordinates.size) {
        val cur = coordinates[i]
        if (cur[0] > prev[0] && cur[1] > prev[1]) {
            prev = cur
            c++
        }
    }

    return c
}

fun dfs(at: Int, k: Int, nums: IntArray, or: Int, ors: Array<MutableList<Int>>) {
    if (k == 0) {
//        println("At : ${at - 1}, Value : $or, At Index ${at - 1}")
        ors[at - 1].add(or)
        return
    }

    for (i in at until nums.size) {
        dfs(i + 1, k - 1, nums, or or nums[i], ors)
    }
}

fun maxMoves(kx: Int, ky: Int, positions: Array<IntArray>): Int {

    println(
        maxMoves(
            0, 1,
            arrayOf(
                intArrayOf(0, 0), intArrayOf(1, 1), intArrayOf(2, 2)
            )
        )
    )

    val minMoves = Array(positions.size) {
        IntArray(positions.size) { Int.MAX_VALUE }
    }

    val posMapping = hashMapOf<String, Int>()
    for (i in 0 until positions.size) {
        val (x, y) = positions[i]
        val key = "$x|$y"
        posMapping[key] = i
    }

    for (i in 0 until positions.size) {
        val (x, y) = positions[i]
        bfs(i, x, y, minMoves, posMapping)
    }

    val initials = IntArray(positions.size) { Int.MAX_VALUE }
    bfsSecond(kx, ky, initials, posMapping)

    val allBitSet = (1 shl positions.size) - 1
    var max = 0
    for (i in 0 until initials.size) {
        val setBit = 1 shl i
        val moves = initials[i]
        val res = moves + dpExplore(0, i, setBit, posMapping, allBitSet, minMoves)
        max = Math.max(max, res)
        break
    }

    return max
}

fun dpExplore(
    isAlice: Int, at: Int, captured: Int, posMapping: HashMap<String, Int>, allBitSet: Int, minMoves: Array<IntArray>
): Int {
    if (captured == allBitSet) {
        return 0
    }

    if (isAlice == 1) {
        var max = 0
        var movesBit = 0
        for (bit in 0 until minMoves[at].size) {
            if ((1 shl bit) and captured == 0) {
                if (minMoves[at][bit] > max) {
                    max = minMoves[at][bit]
                    movesBit = bit
                }
            }
        }
        val setBit = (1 shl movesBit) or captured
        return max + dpExplore(0, movesBit, setBit, posMapping, allBitSet, minMoves)
    }

    var min = Int.MAX_VALUE
    var movesBit = 0
    for (bit in 0 until minMoves[at].size) {
        if ((1 shl bit) and captured == 0) {
            if (minMoves[at][bit] < min) {
                min = minMoves[at][bit]
                movesBit = bit
            }
        }
    }
    val setBit = (1 shl movesBit) or captured
    return min + dpExplore(1, movesBit, setBit, posMapping, allBitSet, minMoves)
}

fun bfsSecond(i: Int, j: Int, minMoves: IntArray, posMapping: HashMap<String, Int>) {
    val queue = LinkedList<Moves>()
    queue.add(Moves(i, j, 0))
    val visited = mutableSetOf<String>()
    visited.add("$i|$j")
    while (queue.isNotEmpty()) {
        for (k in 0 until queue.size) {
            val (x, y, mov) = queue.poll()
            val key = "$x|$y"
            if (posMapping.containsKey(key)) {
                val pos = posMapping[key]!!
                minMoves[pos] = mov
            }
            for (dir in dirs) {
                val nx = x + dir[0]
                val ny = y + dir[1]
                if (nx in 0..49 && ny in 0..49 && !visited.contains("$nx|$ny")) {
                    visited.add("$nx|$ny")
                    queue.add(Moves(nx, ny, mov + 1))
                }
            }
        }
    }
}

fun bfs(at: Int, i: Int, j: Int, minMoves: Array<IntArray>, posMapping: HashMap<String, Int>) {
    val queue = LinkedList<Moves>()
    queue.add(Moves(i, j, 0))
    val visited = mutableSetOf<String>()
    visited.add("$i|$j")
    while (queue.isNotEmpty()) {
        for (i in 0 until queue.size) {
            val (x, y, mov) = queue.poll()
            val key = "$x|$y"
            if (posMapping.containsKey(key)) {
                val pos = posMapping[key]!!
                minMoves[at][pos] = mov
            }
            for (dir in dirs) {
                val nx = x + dir[0]
                val ny = y + dir[1]
                if (nx in 0..49 && ny in 0..49 && !visited.contains("$nx|$ny")) {
                    visited.add("$nx|$ny")
                    queue.add(Moves(nx, ny, mov + 1))
                }
            }
        }
    }
}

data class Moves(val x: Int, val y: Int, val moves: Int)

fun minValidStrings(words: Array<String>, target: String): Int {

    println(
        minValidStrings(
            arrayOf(
                "abc", "def"
            ), "abcdef"
        )
    )

    val root = Trie()
    for (w in words) {
        var cur = root
        for (i in 0 until w.length) {
            val char = w[i]
            if (cur.children[char - 'a'] == null) {
                cur.children[char - 'a'] = Trie()
            }
            cur = cur.children[char - 'a']!!
        }
    }

    val memo = IntArray(words.size) { -1 }
    val res = dpExplore(0, target, root, memo)
    if (res == Int.MAX_VALUE) return -1

    return res
}

fun dpExplore(at: Int, target: String, root: Trie, memo: IntArray): Int {
    if (at == target.length) return 0

    if (memo[at] != -1) return memo[at]

    var cur = root
    var min = Int.MAX_VALUE
    var max = -1
    for (i in at until target.length) {
        val char = target[i]
        if (cur.children[char - 'a'] != null) {
            max = i
            cur = cur.children[char - 'a']!!
        } else break
    }

    for (j in max downTo at) {
        val res = dpExplore(j + 1, target, root, memo)
        if (res != Int.MAX_VALUE) {
            min = Math.min(min, res + 1)
        }
    }

    memo[at] = min
    return min
}

class Trie {
    val children = Array<Trie?>(26) { null }
}

fun maxScore(a: IntArray, b: IntArray): Long {

    println(
        maxScore(
            intArrayOf(-1, 4, 5, -2), intArrayOf(-5, -1, -3, -2, -4)
        )
    )

    val memo = hashMapOf<String, Long>()
    return dpExplore(0, 0, a, b, memo)
}

fun dpExplore(aAt: Int, bAt: Int, a: IntArray, b: IntArray, memo: HashMap<String, Long>): Long {
    if (aAt == a.size) return 0L

    if (bAt == b.size) {
        return Long.MIN_VALUE
    }

    val key = "$aAt|$bAt"
    if (memo.containsKey(key)) return memo[key]!!
    val score = a[aAt] * b[bAt].toLong()
    var take = dpExplore(aAt + 1, bAt + 1, a, b, memo)
    if (take != Long.MIN_VALUE) {
        take += score
    }
    val noTake = dpExplore(aAt, bAt + 1, a, b, memo)

    val res = Math.max(take, noTake)
    memo[key] = res
    return res
}

fun getSneakyNumbers(nums: IntArray): IntArray {

    println(
        getSneakyNumbers(
            intArrayOf(
                7, 1, 5, 4, 3, 4, 6, 0, 9, 5, 8, 2
            )
        )
    )

    val freq = hashMapOf<Int, Int>()
    for (n in nums) {
        freq[n] = (freq[n] ?: 0) + 1
    }

    val ans = mutableListOf<Int>()
    for ((k, v) in freq) {
        if (v == 2) ans.add(k)
    }

    return ans.toIntArray()
}

fun dfsOr(at: Int, nums: IntArray, k: Int, res: Int) {
    if (at >= nums.size - k + 1) return

    if (k == 0) {
        println("At : ${at - 1} Or Value : $res")
        return
    }

    dfsOr(at + 1, nums, k - 1, res or nums[at])
    dfsOr(at + 1, nums, k, res)
}

fun findSafeWalk(grid: List<List<Int>>, health: Int): Boolean {

    println(
        findSafeWalk(
            listOf(
                listOf(0, 1, 0, 0, 0), listOf(0, 1, 0, 1, 1), listOf(0, 0, 0, 1, 1)
            ), 10
        )
    )

    val queue = LinkedList<PosWalk>()
    val memo = Array(grid.size) { IntArray(grid[0].size) { -1 } }
    var health = health
    if (grid[0][0] == 1) health--
    queue.add(PosWalk(0, 0, health))
    memo[0][0] = health
    while (queue.isNotEmpty()) {
        val p = queue.poll()
        for (dir in dirs) {
            var remain = p.health
            val i = p.i + dir[0]
            val j = p.j + dir[1]
            if (i < 0 || j < 0 || i >= grid.size || j >= grid[0].size) continue
            if (grid[i][j] == 1) --remain
            if (remain > memo[i][j] && remain > 0) {
                queue.add(PosWalk(i, j, remain))
                memo[i][j] = remain
            }
        }
    }

    return grid[grid.size - 1][grid[0].size - 1] > 0
}

class PosWalk(val i: Int, val j: Int, val health: Int)

fun maxPossibleScore(start: IntArray, d: Int): Int {

    println(
        maxPossibleScore(
            intArrayOf(2, 6, 13, 13), 5
        )
    )

    start.sort()
    var ans = 0
    var l = 0
    var r = Int.MAX_VALUE

    while (l <= r) {
        val mid = l + (r - l) / 2
        if (inRange(mid, start, d)) {
            ans = Math.max(ans, mid)
            l = mid + 1
        } else {
            r = mid - 1
        }
    }

    return ans
}

fun inRange(mid: Int, nums: IntArray, d: Int): Boolean {

    var prev = nums[0]
    for (i in 1 until nums.size) {
        val l = nums[i]
        val r = nums[i] + d
        val max = Math.max(prev + mid, l)
        if (max !in l..r) {
            return false
        }
        prev = max
    }

    return true
}

fun findMaximumScore(nums: List<Int>): Long {

    val list = mutableListOf<Pair<Int, Int>>()
    list.add(nums[0] to 0)
    for (i in 1 until nums.size) {
        if (list[list.size - 1].first < nums[i]) {
            list.add(nums[i] to i)
        }
    }

    var ans = 0L
    for (i in 0 until list.size - 1) {
        val prev = list[i]
        val next = list[i + 1]
        ans += (prev.first) * (next.second - prev.second)
    }

    val last = list[list.size - 1]
    ans += (last.first * (nums.size - 1 - last.second))

    return ans
}

fun convertDateToBinary(date: String): String {
    return date.split("-").map { it.toInt() }.map { Integer.toBinaryString(it) }.joinToString("-")
}

fun minChanges(nums: IntArray, k: Int): Int {

    println(
        minChanges(
            intArrayOf(3, 4, 5, 2, 1, 7, 3, 4, 7), 3
        )
    )

    return 0
}

fun maximumSubarrayXor(nums: IntArray, queries: Array<IntArray>): IntArray {

    println(
        maximumSubarrayXor(
            intArrayOf(2, 8, 4, 32, 16, 1), arrayOf(
                intArrayOf(0, 2), intArrayOf(1, 4), intArrayOf(0, 5)
            )
        )
    )

    val dp = Array(nums.size) { Array(nums.size) { 0 to 0 } }
    for (i in 0 until nums.size) {
        dp[i][i] = Pair(nums[i], nums[i])
    }

    for (j in 1 until nums.size) {
        for (i in 0 until nums.size) {
            if (i + j < nums.size) {
                val left = dp[i + 1][i + j]
                val bottom = dp[i][i + j - 1]
                val xor = left.first xor bottom.first
                val max = Math.max(left.second, Math.max(bottom.second, xor))
                dp[i][i + j] = xor to max
            }
        }
    }

    val ans = IntArray(queries.size) { 0 }
    for (i in 0 until queries.size) {
        val (l, r) = queries[i]
        ans[i] = dp[l][r].second
    }

    return ans
}

fun wordBreak(s: String, wordDict: List<String>): Boolean {

    println(
        minDamage(
            62, intArrayOf(80, 79), intArrayOf(86, 13)
        )
    )

    return false
}

fun minDamage(power: Int, damage: IntArray, health: IntArray): Long {

    val pairs = mutableListOf<Pair<Int, Int>>()
    for (i in 0 until damage.size) {
        pairs.add(Pair(damage[i], health[i]))
    }

    pairs.sortWith { o1, o2 ->
        if (o1.first == o2.first) {
            o1.second - o2.second
        }
        o2.first - o1.first
    }

    var ans = 0L
    var allDamage = 0
    for ((dam, _) in pairs) {
        allDamage += dam
    }

    for ((dam, heal) in pairs) {
        val seconds = Math.ceil(heal / power.toDouble()).toInt()
        ans += (seconds * allDamage).toLong()
        allDamage -= dam
    }

    return ans
}

fun countGoodIntegers(n: Int, k: Int): Long {


    return 0L
}

fun stringHash(s: String, k: Int): String {

    val ans = StringBuilder()
    for (i in 0..s.length - k step k) {
        val hash = s.substring(i, i + k).map { it - 'a' }.sum() % 26
        val char = hash + 'a'.code
        ans.append(char)
    }

    return ans.toString()
}

fun generateKey(num1: Int, num2: Int, num3: Int): Int {
    val list = listOf(num1, num2, num3).map { addLeadingZero(it) }
    val ans = StringBuilder()
    for (i in 0 until 4) {
        var max = 10
        for (j in 0 until list.size) {
            max = Math.max(max, list[j][i] - '0')
        }
        ans.append(max)
    }

    return ans.toString().toInt()
}

fun addLeadingZero(n1: Int): String {
    var str = "$n1"
    while (str.length < 4) {
        str = "0$str"
    }

    return str
}

class Solution {

    fun minimumCost(target: String, words: Array<String>, costs: IntArray): Int {
        val trie = Trie(words, costs)
        val dp = IntArray(target.length + 1) { Int.MAX_VALUE }
        dp[0] = 0

        fun dfs(index: Int): Int {
            if (index >= target.length) return 0
            if (dp[index] != Int.MAX_VALUE) return dp[index]

            var minCost = Int.MAX_VALUE
            for (j in trie.suffixesAfterAppending(target[index])) {
                val word = words[j]
                if (index + word.length <= target.length && target.startsWith(word, index)) {
                    val cost = costs[j] + dfs(index + word.length)
                    minCost = minOf(minCost, cost)
                }
            }
            dp[index] = minCost
            return minCost
        }

        val result = dfs(0)
        return if (result == Int.MAX_VALUE) -1 else result
    }

    class Trie(words: Array<String>, costs: IntArray) {
        private val root = TrieNode()
        private var currentNode: TrieNode = root

        init {
            for (i in words.indices) {
                val word = words[i]
                var node = root
                for (char in word) {
                    if (node.children[char] == null) {
                        node.children[char] = TrieNode()
                    }
                    node = node.children[char]!!
                }
                node.wordId = i
                node.cost = costs[i]
            }

            buildSuffixLinks()
        }

        private fun buildSuffixLinks() {
            val queue = LinkedList<TrieNode>()
            root.suffixLink = root
            root.children.values.forEach {
                it.suffixLink = root
                queue.add(it)
            }

            while (queue.isNotEmpty()) {
                val node = queue.poll()
                node.children.forEach { (char, child) ->
                    var suffix = node.suffixLink
                    while (suffix != root && suffix?.children?.get(char) == null) {
                        suffix = suffix?.suffixLink
                    }
                    child.suffixLink = suffix.children[char] ?: root
                    child.dictLink =
                        child.suffixLink?.let { suffix -> if (suffix.wordId >= 0) suffix else suffix.dictLink } ?: root
                    queue.add(child)
                }
            }
        }

        fun suffixesAfterAppending(char: Char): List<Int> {
            while (currentNode != root && currentNode.children[char] == null) {
                currentNode = currentNode.suffixLink!!
            }
            if (currentNode.children[char] != null) {
                currentNode = currentNode.children[char]!!
            } else {
                currentNode = root
            }
            return generateSequence(currentNode) { it.dictLink }.mapNotNull { it.wordId }.toList()
        }

        data class TrieNode(
            val children: MutableMap<Char, TrieNode> = mutableMapOf(),
            var suffixLink: TrieNode? = null,
            var dictLink: TrieNode? = null,
            var wordId: Int = -1,
            var cost: Int = 0
        )
    }
}

fun maxScore(grid: List<List<Int>>): Int {

    val list = mutableListOf<Pair<Int, Int>>()
    for (i in 0 until grid.size) {
        for (j in 0 until grid[i].size) {
            list.add(Pair(grid[i][j], i))
        }
    }

    list.sortBy { it.first }

    println(list)

    val memo = Array(list.size + 1) { IntArray(1 shl 11) { -1 } }
    return dpExplore(0, 0, list, memo)
}

fun dpExplore(at: Int, bit: Int, list: List<Pair<Int, Int>>, memo: Array<IntArray>): Int {
    if (at == list.size) return 0

    if (memo[at][bit] != -1) return memo[at][bit]

    val at1 = at

    var max = dpExplore(at + 1, bit, list, memo)
    var at = at
    while (at < list.size && bit and (1 shl list[at].second) != 0) {
        at++
    }

    if (at == list.size) return 0
    val newBit = bit or (1 shl list[at].second)
    val cur = list[at]
    while (at < list.size && list[at].first == cur.first) {
        at++
    }

    val res = cur.first + dpExplore(at, newBit, list, memo)
    max = Math.max(max, res)
    memo[at1][bit] = max
    return max
}

fun resultsArray(queries: Array<IntArray>, k: Int): IntArray {

    val queue = PriorityQueue<Int>(Collections.reverseOrder())
    val first = queries[0]
    val dist = Math.abs(first[0]) + Math.abs(first[1])

    val ans = IntArray(queries.size) { -1 }
    if (k == 1) {
        ans[0] = dist
    }
    queue.add(dist)
    for (i in 1 until queries.size) {
        val (x, y) = queries[i]
        val cur = Math.abs(x) + Math.abs(y)
        val max = queue.peek()
        if (queue.size < k) {
            queue.add(cur)
        } else if (cur < max) {
            queue.poll()
            queue.add(cur)
        }

        if (queue.size == k) {
            ans[i] = queue.peek()
        }
    }

    return ans
}




























