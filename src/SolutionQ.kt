import java.util.PriorityQueue

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




fun findCriticalAndPseudoCriticalEdges(n: Int, edges: Array<IntArray>): List<List<Int>> {

    println(
        findCriticalAndPseudoCriticalEdges(
            5,
            arrayOf(
                intArrayOf(0, 1, 1),
                intArrayOf(1, 2, 1),
                intArrayOf(2, 3, 2),
                intArrayOf(0, 3, 2),
                intArrayOf(0, 4, 3),
                intArrayOf(3, 4, 3),
                intArrayOf(1, 4, 6)
            )
        )
    )

    val adj = Array(n) { mutableListOf<Edge>() }
    for (i in 0 until edges.size) {
        val (u, v, w) = edges[i]
        adj[u].add(Edge(v, w, i))
        adj[v].add(Edge(u, w, i))
    }

    val atLeastOneMST = mutableSetOf<Int>()
    val (mst, result) = findMST(adj)
    atLeastOneMST.addAll(result)
    val ans = List(2) { mutableListOf<Int>() }

    for (i in 0 until edges.size) {
        val (curMst, curMstEdges) = findMSTWithRestrict(adj, i)
        if (curMst == -1 || curMst > mst) {
            ans[0].add(i)
        } else if (curMst == mst) {
            atLeastOneMST.addAll(curMstEdges)
            ans[1].add(i)
        }
    }

    val allMstEdges = mutableSetOf<Int>()
    for (neigh in ans[1]) {
        if (atLeastOneMST.contains(neigh)) {
            allMstEdges.add(neigh)
        }
    }

    ans[1].clear()
    ans[1].addAll(allMstEdges.toList())

    return ans
}

private fun findMSTWithRestrict(adj: Array<MutableList<Edge>>, excludeEdgePos: Int): Result {

    val visited = BooleanArray(adj.size) { false }
    val queue = PriorityQueue(object : Comparator<List<Int>> {
        override fun compare(o1: List<Int>, o2: List<Int>): Int {
            return o1[1] - o2[1]
        }
    })

    val edges = mutableSetOf<Int>()
    var sum = 0
    queue.add(listOf(0, 0, -1))
    while (queue.isNotEmpty()) {
        val (node, weight, edgePos) = queue.poll()
        if (!visited[node]) {
            visited[node] = true
            sum += weight
            edges.add(edgePos)
            for (neigh in adj[node]) {
                if (excludeEdgePos != neigh.edgePos)
                    queue.add(listOf(neigh.dest, neigh.weight, neigh.edgePos))
            }
        }
    }

    for (v in visited)
        if (!v)
            return Result(-1, edges)

    return Result(sum, edges)
}

private fun findMST(adj: Array<MutableList<Edge>>): Result {

    val visited = BooleanArray(adj.size) { false }
    val queue = PriorityQueue(object : Comparator<List<Int>> {
        override fun compare(o1: List<Int>, o2: List<Int>): Int {
            return o1[1] - o2[1]
        }
    })

    val mstEdges = mutableSetOf<Int>()
    var sum = 0
    queue.add(listOf(0, 0, -1))
    while (queue.isNotEmpty()) {
        val (node, weight, edgePos) = queue.poll()
        if (!visited[node]) {
            visited[node] = true
            sum += weight
            mstEdges.add(edgePos)
            for (neigh in adj[node]) {
                queue.add(listOf(neigh.dest, neigh.weight, neigh.edgePos))
            }
        }
    }

    return Result(sum, mstEdges)
}

data class Edge(val dest: Int, val weight: Int, val edgePos: Int)

data class Result(val mst: Int, val edges: Set<Int>)

fun countBlackBlocks(m: Int, n: Int, coordinates: Array<IntArray>): LongArray {

    println(
        countBlackBlocks(
            3, 3, arrayOf(
                intArrayOf(0, 0),
                intArrayOf(1, 1),
                intArrayOf(0, 2)
            )
        )
    )

    val ans = LongArray(5) { 0L }
    ans[0] = (m - 1L) * (n - 1L)

    val sorted = coordinates.sortedWith(object : Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            if (o1[0] == o2[0]) return o1[1] - o2[1]
            return o1[0] - o2[0]
        }
    })

    val blacks = sorted.map { Black(it[0], it[1]) }.toSet()
    val corners = mutableSetOf<Black>()

    val dirs = arrayOf(
        intArrayOf(-1, 0),
        intArrayOf(0, -1),
        intArrayOf(0, 1),
        intArrayOf(1, 0)
    )

    for ((i, j) in coordinates) {
        corners.add(Black(i, j))
        for (dir in dirs) {
            val nI = dir[0] + i
            val nJ = dir[1] + j
            if (nI < 0 || nI >= m || nJ < 0 || nJ >= n) continue
            corners.add(Black(nI, nJ))
        }
    }

    corners.sortedWith(object : Comparator<Black> {
        override fun compare(o1: Black, o2: Black): Int {
            if (o1.i == o2.i) return o1.j - o2.j
            return o1.i - o2.i
        }
    })

    for ((i, j) in corners) {
        if (i < m - 1 && j < n - 1) {
            val count = countBlacks(i, j, blacks)
            ans[0]--
            ans[count]++
        }
    }

    return ans
}

fun countBlacks(i: Int, j: Int, blacks: Set<Black>): Int {
    var c = if (blacks.contains(Black(i, j))) 1 else 0
    val dir = arrayOf(
        intArrayOf(1, 0),
        intArrayOf(0, 1),
        intArrayOf(1, 1),
    )

    dir.forEach {
        val cur = Black(i + it[0], j + it[1])
        if (blacks.contains(cur)) c++
    }

    return c
}

data class Black(val i: Int, val j: Int)

fun findMSTSum(): Int {


    return 0
}

fun canArrange(arr: IntArray, k: Int): Boolean {

    println(
        canArrange(
            intArrayOf(-1, 1, -2, 2, -3, 3, -4, 4),
            3
        )
    )

    val freq = hashMapOf<Int, Int>()
    for (a in arr) {
        val mod = a % k
        freq[mod] = freq.getOrDefault(mod, 0) + 1
    }

    val keys = freq.keys.toSet().sorted().filter { it != 0 }
    if ((freq[0] ?: 0) % 2 == 1) {
        return false
    }

    val used = mutableSetOf<Int>()
    for (key in keys) {
        if (used.contains(key))
            continue
        if (key > 0) {
            val fr1 = freq[key]
            val fr2 = freq[k - key]
            used.add(k - key)
            if (fr1 != fr2)
                return false
        } else {
            val fr1 = freq[key]
            val fr2 = freq[Math.abs(key)]
            used.add(Math.abs(key))
            if (fr1 != fr2)
                return false
        }
    }

    return true
}

fun minInteger(num: String, k: Int): String {

    println(
        minInteger(
            "4321", 4
        )
    )

    val nums = num.map { it - '0' }.toIntArray()
    var k = k
    for (left in 0 until nums.size) {
        val right = left + Math.min(k, nums.size - 1 - left)
        k -= changeMinInRange(left, right, nums)
    }

    val ans = StringBuilder()
    for (n in num) {
        ans.append(n)
    }

    return num
}

fun changeMinInRange(left: Int, right: Int, nums: IntArray): Int {

    var minNum = nums[left]
    var minPos = left
    for (j in left..right) {
        if (nums[j] < minNum) {
            minNum = nums[j]
            minPos = j
        }
    }

    if (minNum != nums[left]) {
        val target = nums[minPos]
        for (i in minPos - 1 downTo left) {
            nums[i + 1] = nums[i]
        }
        nums[left] = target
        return minPos - left
    }

    return 0
}

fun minimumOperations(nums: IntArray, target: IntArray): Long {

    println(
        minimumOperations(
            intArrayOf(3, 5, 1, 2), intArrayOf(4, 6, 2, 4)
        )
    )

    val dif = LongArray(nums.size)
    for (i in 0 until nums.size) {
        val cur = nums[i] - target[i].toLong()
        dif[i] = cur
    }

    var ans = 0L
    for (i in 1 until dif.size) {
        val prev = dif[i - 1]
        val cur = dif[i]
        if (prev > 0 && cur > 0) {
            if (prev > cur) {
                ans += prev - cur
            }
        } else if (prev < 0 && cur < 0) {
            if (prev < cur) {
                ans += Math.abs(prev - cur)
            }
        } else {
            ans += Math.abs(prev)
        }
    }

    ans += Math.abs(dif[dif.size - 1])

    return ans
}


fun maxOperations(s: String): Int {

    println(
        maxOperations(
            "1001101"
        )
    )

    val nums = s.map { it - '0' }.toIntArray()
    var ones = 0
    var ans = 0
    for (i in 0 until nums.size - 1) {
        if (nums[i] == 1) {
            ones++
        }
        if (i > 0 && nums[i] == 0 && nums[i - 1] == 1) {
            ans += ones
        }
    }

    return ans
}

fun doesAliceWin(s: String): Boolean {

    var c = 0
    val vowels = setOf('a', 'e', 'i', 'o', 'u')
    for (ch in s) {
        if (vowels.contains(ch)) c++
    }

    return c != 0
}


fun countPairs(coordinates: List<List<Int>>, k: Int): Int {


    return k
}
fun minimumLength(s: String): Int {

    val freq = hashMapOf<Char, Int>()
    for (c in s) {
        freq[c] = freq.getOrDefault(c, 0) + 1
        if (freq[c]!! >= 3) {
            freq[c] = freq.getOrDefault(c, 0) - 2
        }
    }

    val c = freq.values.sum()
    return c
}

fun losingPlayer(x: Int, y: Int): String {

    //115 + 75 + 10
    var x75 = x
    var y10 = y
    var isAlice = true

    while (true) {
        x75 -= 1
        y10 -= 4
        if (x75 < 0 || y10 < 0) {
            if (!isAlice) return "Alice"
            else return "Bob"
        }
        isAlice = !isAlice
    }

    return "Alice"
}

fun isTransformable(s: String, t: String): Boolean {

    println(
        isTransformable(
            "34521", "23415"
        )
    )

    val nums = s.map { it - '0' }.toIntArray()
    val targets = t.map { it - '0' }.toIntArray()

    for (i in t.length - 1 downTo 0) {
        val cur = nums[i]
        val tar = targets[i]
        if (cur > tar) return false
        else if (cur != tar) {
            var targetIndex = -1
            for (l in i downTo 0) {
                if (nums[l] == tar) {
                    targetIndex = l
                    break
                }
            }
            if (targetIndex == -1) return false
            sortInRange(targetIndex, i, nums)
        }

    }

    return true
}

fun sortInRange(l: Int, r: Int, nums: IntArray) {
    val list = nums.toList().subList(l, r + 1).toMutableList()
    list.sort()
    for (i in l..r) {
        nums[i] = list[i - l]
    }
}

fun numWays(words: Array<String>, target: String): Int {
    println(
        numWays(
            arrayOf("abba", "baab"), "bab"
        )
    )
    return dpExplore(0, 0, 0, words, target).toInt()
}

fun dpExplore(r: Int, c: Int, atTarget: Int, words: Array<String>, target: String): Long {
    if (atTarget == target.length) return 1L

    if (r == words.size) return 0

    var ans = dpExplore(r + 1, c, atTarget, words, target)
    for (curC in c until words[r].length) {
        if (words[r][curC] == target[atTarget]) {
            ans += dpExplore(r, curC + 1, atTarget + 1, words, target)
            ans += dpExplore(r + 1, curC + 1, atTarget + 1, words, target)
        }
    }

    return ans
}

fun connectTwoGroups(cost: List<List<Int>>): Int {

    println(
        connectTwoGroups(
            listOf(
                listOf(1, 3), listOf(4, 1)
            )
        )
    )

    val memo = Array(cost.size + 1) { IntArray((1 shl cost[0].size) - 1) { -1 } }
    return solve(0, 0, cost, memo)
}

fun solve(at: Int, visited: Int, cost: List<List<Int>>, memo: Array<IntArray>): Int {
    if (at == cost.size && visited == (1 shl cost[0].size) - 1) {
        return 0
    }

    if (memo[at][visited] != -1) return memo[at][visited]

    var min = Int.MAX_VALUE
    if (at == cost.size) {
        var unSetBit = -1
        for (j in 0 until cost[0].size) {
            if (visited and (1 shl j) == 0) {
                unSetBit = j
                break
            }
        }

        for (r in 0 until cost.size) {
            val res = cost[r][unSetBit] + solve(at, visited or (1 shl unSetBit), cost, memo)
            min = Math.min(min, res)
        }
    } else {
        for (j in 0 until cost[0].size) {
            val res = cost[at][j] + solve(at + 1, visited or (1 shl j), cost, memo)
            min = Math.min(min, res)
        }
    }

    memo[at][visited] = min
    return min
}

