import utils.aho.AhoCorasick
import java.util.*
import kotlin.Comparator
import kotlin.collections.HashMap
import kotlin.math.cos
import kotlin.math.min

private val mod = Math.pow(10.0, 9.0).toInt() + 7
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

    println(
        minimumCost(
            "abc",
            arrayOf("a", "b", "c", "ab"),
            intArrayOf(3, 14, 10, 1)
        )
    )

}

fun minimumCost(target: String, words: Array<String>, costs: IntArray): Int {

    val aho = AhoCorasick()
    for (i in 0 until words.size) {
        aho.addPattern(words[i], costs[i])
    }

    aho.buildFailureLink()
    val resAho = aho.search(target)
    val strMap = hashMapOf<Int, MutableList<AhoCorasick.StringPos>>()

    for (i in resAho) {
        if (strMap[i.start] == null)
            strMap[i.start] = mutableListOf()
        strMap[i.start]!!.add(i)
    }

    val memo = HashMap<Int, Int>()
    val res = dpExplore(0, target, strMap, memo)
    if (res == Int.MAX_VALUE)
        return -1
    return res
}

fun dpExplore(
    at: Int,
    target: String,
    strMap: HashMap<Int, MutableList<AhoCorasick.StringPos>>,
    memo: HashMap<Int, Int>
): Int {
    if (at >= target.length)
        return 0
    if (memo.containsKey(at))
        return memo[at]!!
    var min = Int.MAX_VALUE

    for (next in strMap[at] ?: emptyList()) {
        val res = dpExplore(next.start + next.str.length, target, strMap, memo)
        if (res != Int.MAX_VALUE) {
            min = min(min, res + next.cost)
        }
    }

    memo[at] = min
    return min
}

fun dpExplore(prevPrev: Int, prev: Int, at: Int, arr: IntArray): Int {
    if (at == arr.size)
        return 0

    if (prev == 1 && prevPrev == 1) {
        return arr[at] + dpExplore(1, 0, at + 1, arr)
    } else if (prevPrev == 0 && prev == 1) {
        var take = arr[at] + dpExplore(1, 0, at + 1, arr)
        if (at - 1 >= 0) {
            take += arr[at - 1]
        }

        var skip = dpExplore(1, 1, at + 1, arr)

        return maxOf(take)
    }

    return 0
}

fun maximumScore(grid: Array<IntArray>): Long {

    println(
        maximumScore(
            arrayOf(
                intArrayOf(0, 0, 0, 0, 0),
                intArrayOf(0, 0, 3, 0, 0),
                intArrayOf(0, 1, 0, 0, 0),
                intArrayOf(5, 0, 0, 3, 0),
                intArrayOf(0, 0, 0, 0, 2)
            )
        )
    )

    return 0L
}

fun nonSpecialCount(l: Int, r: Int): Int {

    var all = r - l + 1
    var i = 3
    if (4 in l..r) {
        all--
    }

    while (true) {
        val square = i * i
        if (square > r) {
            break
        } else if (isPrime(i) && square in l..r) {
            all--
        }
        i++
    }

    return all
}

private fun isPrime(n: Int): Boolean {
    if (n <= 1) return false
    if (n == 2) return true
    if (n % 2 == 0) return false

    val sqrt = Math.sqrt(n.toDouble()).toInt()
    for (d in 3..sqrt) {
        if (n % d == 0) return false
    }
    return true
}

fun numberOfSubstrings(s: String): Int {

    println(numberOfSubstrings("101101"))

    return 0
}


fun minFlips(grid: Array<IntArray>): Int {

    println(
        minFlips(
            arrayOf(
                intArrayOf(0, 0, 0),
                intArrayOf(0, 1, 0),
                intArrayOf(0, 0, 0)
            )
        )
    )

    var change = 0
    var one = 0
    val list = mutableListOf<Pair<Int, Int>>()
    for (i in 0..grid.size / 2) {
        for (j in 0..grid.size / 2) {
            val (avai, ones) = countOnes(i, j, grid)
            list.add(Pair(avai, ones))
            val zeroes = avai - ones
            if (zeroes <= ones) {
                one += avai
            }

            change += Math.min(one, zeroes)
        }
    }

    val mod = one % 4
    if (one % 4 == 0 || mod == 2)
        return change

    if (mod == 1) {
        return change + 1
    }

    println("Ones : $one Change : $change")
    return 0
}

private fun countOnes(i: Int, j: Int, grid: Array<IntArray>): IntArray {

    val row = grid.size - 1
    val col = grid[0].size - 1
    var ones = grid[i][j]
    var avai = 1
    if (col - j > j) {
        avai++
        ones += grid[i][col - j]
    }

    if (row - i > i) {
        avai++
        ones += grid[row - i][j]
    }

    if (row - i > i && col - j > j) {
        avai++
        ones += grid[row - i][col - j]
    }

    return intArrayOf(avai, ones)
}

fun shortestDistanceAfterQueries(n: Int, queries: Array<IntArray>): IntArray {

    println(
        shortestDistanceAfterQueries(
            5,
            arrayOf(
                intArrayOf(2, 4),
                intArrayOf(2, 5),
            )
        )
            .toList()
    )

    var paths = TreeMap<Int, Int>()
    for (i in 0 until n) {
        paths[i] = i + 1
    }

    val ans = IntArray(queries.size) { 0 }
    for (i in 0 until queries.size) {
        val (u, v) = queries[i]
        val newPath = TreeMap<Int, Int>()
        for ((l, r) in paths) {
            if (r <= u) {
                newPath[l] = r
            } else if (v <= l) {
                newPath[l] = r
            }
        }

        if ((paths[u] ?: 0) < v) {
            newPath[u] = v
        } else {
            newPath[u] = paths[u] ?: 0
        }

        paths = newPath
        ans[i] = paths.size
    }

    return ans
}

fun numberOfAlternatingGroups(colors: IntArray, queries: Array<IntArray>): List<Int> {

    println(
        numberOfAlternatingGroups(
            intArrayOf(),
            arrayOf(
                intArrayOf()
            )
        )
    )

    return emptyList()
}

fun countPairs(nums: IntArray): Int {

    println(
        countPairs(
            intArrayOf(1023, 2310, 2130, 213)
        )
    )

    return 0
}

fun getFinalState(nums: IntArray, k: Int, multiplier: Int): IntArray {

    println(
        getFinalState(
            intArrayOf(2, 1, 3, 5, 6),
            5,
            2
        )
            .toList()
    )

    if (multiplier == 1)
        return nums

    val comparator = Comparator<Pair<Long, Int>> { o1, o2 ->
        if (o1.first == o2.first) o1.second - o2.second else o1.first.compareTo(o2.first)
    }

    val queue =
        PriorityQueue(comparator)
    for (i in 0 until nums.size) {
        queue.add(Pair(nums[i].toLong(), i))
    }

    var max = nums.max().toLong()
    var k = k

    while (k > 0) {
        val (curMin, pos) = queue.poll()
        if (curMin * multiplier > max) {
            queue.add(Pair(curMin, pos))
            break
        }
        max = Math.max(max, curMin * multiplier)
        k--
        queue.add(Pair(curMin * multiplier, pos))
    }

    val ans = IntArray(nums.size)
    if (k == 0) {
        for (q in queue) {
            val (min, pos) = q
            ans[pos] = (min % mod).toInt()
        }
        return ans
    }

    val pairs = queue.toTypedArray().sortedWith(comparator).toMutableList()
    val each = k / nums.size

    val curMulti = modExp(multiplier.toLong(), each.toLong(), mod)

    if (each > 0)
        for (i in 0 until pairs.size) {
            val (value, pos) = pairs[i]
            pairs[i] = Pair((value * curMulti) % mod, pos)
        }

    val left = k % nums.size
    for (i in 0 until left) {
        val (value, pos) = pairs[i]
        pairs[i] = Pair((value * multiplier) % mod, pos)
    }

    for (pair in pairs) {
        ans[pair.second] = pair.first.toInt()
    }

    return ans
}

fun modExp(base: Long, exp: Long, mod: Int): Long {
    var result = 1L
    var b = base
    var e = exp

    while (e > 0) {
        if (e % 2 == 1L) {
            result = (result * b) % mod
        }
        b = (b * b) % mod
        e /= 2
    }

    return result
}

fun countPairsBF(nums: IntArray): Int {

    println(
        countPairsBF(
            intArrayOf(8, 12, 5, 5, 14, 3, 12, 3, 3, 6, 8, 20, 14, 3, 8)
        )
    )

    var ans = 0
    for (i in 0 until nums.size) {
        for (j in i + 1 until nums.size) {
            if (checkingEqual(nums[i], nums[j])) {
                ans++
            }
        }
    }

    return ans
}

fun checkingEqual(n1: Int, n2: Int): Boolean {
    return n1 == n2 || everyPairs(n1, n2) || everyPairs(n2, n1)
}

private fun everyPairs(n1: Int, n2: Int): Boolean {

    val str1 = n1.toString()
    for (i in 0 until str1.length) {
        for (j in i + 1 until str1.length) {
            if (str1[i] != str1[j]) {
                val builder = StringBuilder(str1)
                builder.setCharAt(i, str1[j])
                builder.setCharAt(j, str1[i])
                val curNum = builder.toString().toInt()
                if (curNum == n2) {
                    return true
                }
            }
        }
    }

    return false
}

fun countKConstraintSubstrings(s: String, k: Int, queries: Array<IntArray>): LongArray {

    println(
        countKConstraintSubstrings(
            "010101",
            1,
            arrayOf(
                intArrayOf(0, 5),
                intArrayOf(1, 4),
                intArrayOf(2, 3)
            )
        )
    )

    var zero = 0
    var one = 0
    //[1,4], [2,6], [3,7], [4,8], [5,10], [6,12], [7,14]

    var l = 0
    for (r in 0 until s.length) {
        if (s[r] == '1')
            one++
        else
            zero++

        while (zero > k && one > k) {
            if (s[l] == '1')
                one--
            else
                zero--
            l++
        }

        println("($l,$r)")
    }

    val ans = LongArray(queries.size) { 0L }



    return longArrayOf()
}

fun maximumValueSum(board: Array<IntArray>): Long {

    println(
        maximumValueSum(
            arrayOf(
                intArrayOf(1, 2, 3),
                intArrayOf(4, 5, 6),
                intArrayOf(7, 8, 9)
            )
        )
    )

    var ans = Long.MIN_VALUE
    val rows = Array(board.size) { mutableListOf<PosVal>() }
    for (r in 0 until board.size) {
        val curRow = board[r]
        val list = mutableListOf<PosVal>()
        for (c in 0 until curRow.size) {
            list.add(PosVal(curRow[c], r, c))
        }
        list.sort()
        rows[r] = list.subList(0, Math.min(list.size, 4))
    }

    for (i1 in 0 until rows.size - 2) {
        for (i2 in i1 + 1 until rows.size - 1) {
            for (i3 in i2 + 1 until rows.size) {
                val cur = findMax(
                    rows[i1],
                    rows[i2],
                    rows[i3]
                )
                ans = Math.max(ans, cur)
            }
        }
    }

    return ans
}

fun findMax(r1: List<PosVal>, r2: List<PosVal>, r3: List<PosVal>): Long {

    var ans = Long.MIN_VALUE
    for (c1 in 0 until r1.size) {
        for (c2 in 0 until r2.size) {
            for (c3 in 0 until r3.size) {
                if (r1[c1].j != r2[c2].j && r3[c3].j != r1[c1].j && r3[c3].j != r2[c2].j) {
                    val cur = r1[c1].value.toLong() + r2[c2].value + r3[c3].value
                    ans = Math.max(ans, cur)
                }
            }
        }
    }

    return ans
}

data class PosVal(val value: Int, val i: Int, val j: Int) : Comparable<PosVal> {
    override fun compareTo(other: PosVal): Int {
        return other.value - value
    }
}

fun winningPlayerCount(n: Int, pick: Array<IntArray>): Int {

    val map = Array(n) { hashMapOf<Int, Int>() }
    for ((pl, cl) in pick) {
        val cur = map[pl]
        cur[cl] = (cur[cl] ?: 0) + 1
    }

    var c = 0
    for (i in 0 until map.size) {
        val list = map[i].values.toList()
        if (list.isNotEmpty()) {
            val max = list.max()
            if (max >= i + 1)
                c++
        }
    }

    return c
}


fun maxEnergyBoost(energyDrinkA: IntArray, energyDrinkB: IntArray): Long {

    println(
        maxEnergyBoost(
            intArrayOf(4, 1, 1),
            intArrayOf(1, 1, 3)
        )
    )

    val memo = Array(energyDrinkA.size) { LongArray(2) { -1 } }
    val r1 = dpExplore(0, 0, energyDrinkA, energyDrinkB, memo)
    val r2 = dpExplore(1, 0, energyDrinkA, energyDrinkB, memo)

    return Math.max(r1, r2)
}

private fun dpExplore(prev: Int, at: Int, enA: IntArray, enB: IntArray, memo: Array<LongArray>): Long {
    if (at >= enA.size)
        return 0L

    if (memo[at][prev] != -1L)
        return memo[at][prev]

    var max = 0L
    if (prev == 0) {
        val c1 = dpExplore(prev, at + 1, enA, enB, memo) + enA[at]
        val c2 = dpExplore(1, at + 2, enA, enB, memo) + enA[at]
        val c = Math.max(c1, c2)
        max = Math.max(max, c)
    } else {
        val c1 = dpExplore(prev, at + 1, enA, enB, memo) + enB[at]
        val c2 = dpExplore(0, at + 2, enA, enB, memo) + enB[at]
        val c = Math.max(c1, c2)
        max = Math.max(max, c)
    }

    memo[at][prev] = max
    return max
}

fun resultsArray(nums: IntArray, k: Int): IntArray {
    println(
        resultsArray(
            intArrayOf(3, 2, 3, 2, 3, 2),
            2
        )
    )


    val ans = IntArray(nums.size - k + 1) { -1 }
    var at = 0
    while (at < nums.size) {
        var c = 1
        at++
        while (at < nums.size && nums[at - 1] < nums[at]) {
            c++
            if (c >= k) {
                ans[at - k + 1] = nums[at]
            }
            at++
        }
    }

    return ans
}

fun countOfPairs(nums: IntArray): Int {

    val memo = hashMapOf<String, Long>()
    val max = nums.max()

    return solveDp(0, max, 0, nums, max, memo).toInt()
}

private fun solveDp(prevMax: Int, sMax: Int, at: Int, nums: IntArray, max: Int, memo: HashMap<String, Long>): Long {
    if (at == nums.size) {
        return 1
    }

    val key = "$prevMax|$sMax|$at"
    if (memo.containsKey(key))
        return memo[key]!!

    var counter = 0L
    for (i in prevMax..max) {
        val second = nums[at] - i
        if (second in 0..sMax) {
            counter += solveDp(i, second, at + 1, nums, max, memo)
        }
    }

    counter %= mod
    memo[key] = counter
    return counter
}

fun countGoodNodes(edges: Array<IntArray>): Int {

    println(
        countGoodNodes(
            arrayOf(
                intArrayOf()
            )
        )
    )

    val adj = Array(edges.size + 1) { mutableListOf<Int>() }

    for ((a, b) in edges) {
        adj[a].add(b)
        adj[b].add(a)
    }

    dfsCounter(-1, 0, adj)

    return goods
}

var goods = 0

fun dfsCounter(parent: Int, at: Int, adj: Array<MutableList<Int>>): Int {

    val set = mutableSetOf<Int>()
    var curC = 0

    for (child in adj[at]) {
        if (child != parent) {
            val cur = dfsCounter(at, child, adj)
            curC += cur
            set.add(cur)
        }
    }

    if (set.size <= 1) {
        goods++
    }

    return curC + 1
}

fun finalPositionOfSnake(n: Int, commands: List<String>): Int {

    var i = 0
    var j = 0

    for (c in commands) {
        when (c) {
            "UP" -> {
                i--
            }

            "DOWN" -> {
                i++
            }

            "LEFT" -> {
                j--
            }

            "RIGHT" -> {
                j++
            }
        }
    }

    return (i * n) + j
}


fun maxFrequencyScore(nums: IntArray, k: Long): Int {

    var longest = 0
    nums.sort()

    println(
        maxFrequencyScore(
            intArrayOf(1, 4, 4, 2, 4),
            0
        )
    )

    return 0
}

fun sumOfPower(nums: IntArray, k: Int): Int {
    println(
        sumOfPower(
            intArrayOf(1, 2, 3),
            3
        )
    )
    val memo = Array(nums.size) { LongArray(k + 1) { -1L } }
    return dpExplore(0, 0, nums, k, memo).toInt()
}

private fun dpExplore(at: Int, sum: Int, nums: IntArray, k: Int, memo: Array<LongArray>): Long {
    if (sum == k) {
        return findPowerOfTwoMod(nums.size - at)
    }

    if (at == nums.size || sum > k)
        return 0L

    if (memo[at][sum] != -1L)
        return memo[at][sum]

    val skip = 2L * dpExplore(at + 1, sum, nums, k, memo)
    val add = dpExplore(at + 1, sum + nums[at], nums, k, memo)
    val res = (skip + add) % mod
    memo[at][sum] = res

    return res
}

private fun findPowerOfTwoMod(power: Int): Long {
    if (power == 0)
        return 1
    var res = 1L
    var power = power
    while (power > 0) {
        val minBit = Math.min(power, 31)
        val curResult = 1L shl minBit
        power -= minBit
        res *= curResult
        res %= utils.mod
    }
    return res
}

fun minOrAfterOperations(nums: IntArray, k: Int): Int {

    println(
        minOrAfterOperations(
            intArrayOf(3, 5, 3, 2, 7), 2
        )
    )

    var list = nums.toList()
    var k = k
    var freq = getFreqOfSetBits(nums)
    for (bit in 2 downTo 0) {
        if (list.size != freq[bit] && freq[bit] > 0) {
            if (k > freq[bit]) {
                val stack = Stack<Int>()
                stack.push(list[0])
                for (i in 1 until list.size) {
                    val prevBit = stack.peek() and (1 shl bit) != 0
                    val cur = list[i] and (1 shl bit) != 0
                    if (prevBit xor cur) {
                        stack.push(stack.pop() and list[i])
                        k--
                    } else stack.push(list[i])
                }

                list = stack.toList()
                freq = getFreqOfSetBits(list.toIntArray())
            }
        }
    }

    println(list)

    return 0
}

fun getFreqOfSetBits(arr: IntArray): IntArray {

    val freq = IntArray(32) { 0 }
    for (i in 0 until 32) {
        for (a in arr) {
            if (((1 shl i) and a) != 0) {
                freq[i]++
            }
        }
    }

    return freq
}

fun canAliceWin(nums: IntArray): Boolean {

    var single = 0
    var double = 0
    for (n in nums) {
        if (n <= 9) single += n
        else double += n
    }

    if (single == double) return false

    return true
}

private fun countChanges(
    change: Int,
    nums: IntArray,
    k: Int,
): Int {

    var c = 0
    for (i in 0 until nums.size / 2) {
        val l = nums[i]
        val r = nums[nums.size - 1 - i]
        val cur = shouldChange(l, r, k, change)
        c += cur
    }

    return 0
}

private fun shouldChange(a: Int, b: Int, k: Int, targetDif: Int): Int {

    val abs = Math.abs(a - b)
    if (abs == targetDif) return 0
    val min = Math.min(a, b)
    val max = Math.max(a, b)
    val dif1 = k - min
    if (dif1 >= targetDif || max >= targetDif) return 1

    return 2
}

fun solveProblemYandex(board: List<String>): Int {

    var start = intArrayOf()
    for (i in 0 until board.size) {
        for (j in 0 until board[i].length) {
            if (board[i][j] == 'S') {
                start = intArrayOf(i, j)
                break
            }
        }
    }

    val queue = LinkedList<Node>()
    val visited = Array(board.size) { Array(board.size) { false } }
    visited[start[0]][start[1]] = true
    queue.add(Node(start[0], start[1], 0))
    var c = 0
    val hashes = mutableSetOf<String>()
    while (queue.isNotEmpty()) {
        val size = queue.size
        for (i in 0 until size) {
            val p = queue.poll()
            if (board[p.i][p.j] == 'F') return c
            if (p.type == 0) {
                for (dir in horseDir) {
                    val nI = p.i + dir[0]
                    val nJ = p.j + dir[1]
                    if (nI in 0..board.size - 1 && nJ in 0..board.size - 1) {
                        val curHash = "${p.type}$nI$nJ"
                        if (!hashes.contains(curHash)) {
                            hashes.add(curHash)
                            addToLinkedList(nI, nJ, p.type, board, queue)
                        }
                    }
                }
            } else {
                for (dir in karolDir) {
                    val nI = p.i + dir[0]
                    val nJ = p.j + dir[1]
                    if (nI in 0..board.size - 1 && nJ in 0..board.size - 1) {
                        val curHash = "${p.type}$nI$nJ"
                        if (!hashes.contains(curHash)) {
                            hashes.add(curHash)
                            addToLinkedList(nI, nJ, p.type, board, queue)
                        }
                    }
                }
            }
        }
        c++
    }

    return -1
}

fun addToLinkedList(i: Int, j: Int, type: Int, board: List<String>, queue: LinkedList<Node>) {
    when (board[i][j]) {
        'K' -> {
            queue.add(Node(i, j, 0))
        }

        'G' -> {
            queue.add(Node(i, j, 1))
        }

        else -> {
            queue.add(Node(i, j, type))
        }
    }
}

val horseDir = arrayOf(
    intArrayOf(-2, -1),
    intArrayOf(-2, 1),
    intArrayOf(-1, -2),
    intArrayOf(-1, 2),
    intArrayOf(1, -2),
    intArrayOf(2, -1),
    intArrayOf(1, 2),
    intArrayOf(2, 1)
)

val karolDir = arrayOf(
    intArrayOf(0, 1), intArrayOf(0, -1), intArrayOf(1, 0), intArrayOf(-1, 0),
    intArrayOf(-1, -1), intArrayOf(-1, 1), intArrayOf(1, -1), intArrayOf(1, 1),
)

//0 Horse
//1 Karol
class Node(val i: Int, val j: Int, val type: Int)





