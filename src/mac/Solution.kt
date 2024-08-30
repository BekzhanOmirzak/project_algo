package mac

import DynamicSegmentTree
import test.bs
import java.util.*
import java.util.function.ToIntFunction
import kotlin.collections.HashMap
import kotlin.math.ceil
import kotlin.math.max

fun main() {

    println(
        maxTotalReward(
            intArrayOf(
                1, 5, 4
            )
        )
    )

}



fun maxTotalReward(rewardValues: IntArray): Int {

    rewardValues.sort()
    for (i in 0 until rewardValues.size) {
        dpExplore(i, 0, rewardValues)
    }

    return 0
}


fun dpExplore(at: Int, sum: Int, rewardValues: IntArray): Int {

    return 0
}


fun findBS(left: Int, sum: Int, rewardValues: IntArray): Int {
    var l = left
    var r = rewardValues.size - 1
    var start = r
    while (l <= r) {
        val mid = (l + r) / 2
        if (rewardValues[mid] > sum) {
            r = mid - 1
            start = Math.min(start, mid)
        } else {
            l = mid + 1
        }
    }
    return start
}

fun valueAfterKSeconds(n: Int, k: Int): Int {

    println(
        valueAfterKSeconds(
            5, 3
        )
    )
    val mod = Math.pow(10.0, 9.0).toInt() + 5
    val prefix = LongArray(n) { 1 }
    for (i in 1..k) {
        var sum = 0L
        for (j in 0 until prefix.size) {
            sum += prefix[j] % mod
            prefix[j] = sum % mod
        }
    }

    return (prefix[prefix.size - 1] % mod).toInt()
}

fun numberOfChild(n: Int, k: Int): Int {

    var seconds = 1
    var at = 0
    var increase = true
    while (seconds <= k) {
        if (increase) {
            at++
        } else {
            at--
        }
        seconds++
        if (at == n - 1)
            increase = false
        else if (at == 0) {
            increase = true
        }
    }

    return at
}

fun minimumDifference(nums: IntArray, k: Int): Int {

    println(
        minimumDifference(
            intArrayOf(1, 2, 1, 2),
            2
        )
    )

    return 0
}

fun clearStars(s: String): String {

    val queue = PriorityQueue(object : Comparator<CharPos> {
        override fun compare(o1: CharPos, o2: CharPos): Int {
            if (o1.char == o2.char)
                return o2.pos - o1.pos
            return o1.char - o2.char
        }
    })

    val ans = StringBuilder()

    for (i in 0 until s.length) {
        val char = s[i]
        if (char == '*') {
            queue.poll()
        } else {
            queue.add(CharPos(char, i))
        }
    }

    queue.sortedWith(object : Comparator<CharPos> {
        override fun compare(o1: CharPos, o2: CharPos): Int {
            return o1.pos - o2.pos
        }
    })

    val list = mutableListOf<CharPos>()

    list.addAll(queue)
    list.sortWith(object : Comparator<CharPos> {
        override fun compare(o1: CharPos, o2: CharPos): Int {
            return o1.pos - o2.pos
        }
    })

    for (l in list) {
        ans.append(l.char)
    }

    return ans.toString()
}

class CharPos(
    val char: Char,
    val pos: Int
)

fun countDays(days: Int, meetings: Array<IntArray>): Int {

    println(
        countDays(
            10,
            arrayOf(
                intArrayOf(1, 7),
                intArrayOf(1, 3),
                intArrayOf(9, 9)
            )
        )
    )

    val sorted = meetings.sortedWith(object : Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            if (o1[0] == o2[0])
                return o2[1] - o1[1]
            return o1[0] - o2[0]
        }
    })

    val allMeets = mutableListOf<IntArray>()

    var at = 0
    while (at < sorted.size) {
        var (start, end) = sorted[at]
        at++
        while (at < sorted.size && sorted[at][0] <= end) {
            end = max(end, sorted[at][1])
            at++
        }

        allMeets.add(intArrayOf(start, end))
    }

    allMeets.forEach {
        println(it.toList())
    }

    var ans = allMeets[0][0] - 1 + (days - allMeets[allMeets.size - 1][1])

    for (i in 0 until allMeets.size - 1) {
        val prev = allMeets[i]
        val next = allMeets[i + 1]
        ans += (next[0] - prev[1] - 1)
    }

    return ans
}

fun minimumChairs(s: String): Int {

    var max = 0
    var c = 0
    for (ch in s) {
        if (ch == 'E')
            c++
        else
            c--
        max = max(max, c)
    }

    return max
}

fun minEndSecond(n: Int, x: Int): Long {

    println(
        minEndSecond(
            3,
            4
        )
    )

    if (n == 1)
        return x.toLong()

    val emptyBits = mutableListOf<Int>()
    val list = mutableListOf<Long>()
    list.add(x.toLong())

    for (bit in 0 until 63) {
        if (x and (1 shl bit) == 0) {
            emptyBits.add(bit)
        }
    }

    for (i in 2..n) {
        val curNum = i - 1
        var nextNum = x.toLong()
        var at = 0
        for (bit in 0 until 32) {
            if (curNum and (1 shl bit) != 0) {
                val shouldSetBit = emptyBits[bit]
                nextNum = nextNum or (1L shl shouldSetBit)
                at++
            }
        }
        list.add(nextNum)
    }

    return list[list.size - 1]
}

fun getResults(queries: Array<IntArray>): List<Boolean> {

    println(
        getResults(
            arrayOf(
                intArrayOf(),
                intArrayOf()
            )
        )
    )
    var maxLength = 0

    //saving all obstacles to treeSet
    for (q in queries) {
        if (q[0] == 2) {
            maxLength = Math.max(maxLength, q[1])
        }
    }



    return listOf()
}

class SegmentTree(val size: Int) {

    private val board = IntArray(size + 1) { size }
    private val obstacles = TreeSet<Int>()

    fun insertObstacle(x: Int) {
        if (obstacles.floor(x) != null) {
            val floor = obstacles.floor(x)!!
            board[floor] = x
        }
        if (obstacles.ceiling(x) != null) {
            val ceiling = obstacles.ceiling(x)!!
            board[x] = ceiling
        }
        obstacles.add(x)
    }

    fun findMaxInRange(range: Int): Int {
        return findMaxInRangeHelper(0, 0, size, range)
    }

    private fun findMaxInRangeHelper(at: Int, l: Int, r: Int, range: Int): Int {
        return 0
    }

}


fun numberOfPairs(nums1: IntArray, nums2: IntArray, k: Int): Long {
    println(
        numberOfPairs(
            intArrayOf(1, 3, 4),
            intArrayOf(1, 3, 4),
            1
        )
    )

    val f1 = hashMapOf<Int, Int>()
    val f2 = hashMapOf<Int, Int>()

    for (n in nums1) {
        f1[n] = f1.getOrDefault(n, 0) + 1
    }

    for (n in nums2) {
        f2[n * k] = f2.getOrDefault(n * k, 0) + 1
    }

    val set = setOf(2, 3, 4, 5)

    var ans = 0L
    for ((n, freq) in f1) {
        val sqrt = Math.sqrt(freq.toDouble()).toInt()
        if (f2.containsKey(n)) {
            ans += f2[n]!! * freq
        }

        for (s in set) {
            val multi = sqrt * s
            if (n % multi == 0 && f2.containsKey(multi)) {
                ans += freq * f2[multi]!!
            }
        }

        for (mod in 1..sqrt) {
            if (n % mod == 0 && f2.containsKey(mod)) {
                ans += freq * f2[mod]!!
            }
        }
    }

    return ans
}


fun countBits(n: Int): IntArray {

    println(
        countBits(
            10
        )
    )

    return intArrayOf()
}

fun findProductsOfElements(queries: Array<LongArray>): IntArray {


    return intArrayOf()
}

fun solve(from: Long, to: Long, mod: Long): Int {

    val fromFreq = IntArray(64) { 0 }
    val toFreq = IntArray(64) { 0 }

    for (i in 0..to) {

    }

    return 0
}

fun compressedString(word: String): String {

    val ans = StringBuilder()
    var at = 0
    while (at < word.length) {
        val cur = word[at]
        var c = 0
        while (at < word.length && word[at] == cur) {
            c++
            at++
            if (c == 9) {
                break
            }
        }

        ans.append("$c$cur")
    }

    return ans.toString()
}

fun queryResults(limit: Int, queries: Array<IntArray>): IntArray {

    val ans = IntArray(queries.size) { 0 }
    val colors = hashMapOf<Int, Int>()
    val freq = hashMapOf<Int, Int>()

    for (i in 0 until queries.size) {
        val (pos, cur) = queries[i]
        if (colors.contains(pos)) {
            val prevColor = colors[pos]!!
            if (prevColor != cur) {
                freq[prevColor] = freq.getOrDefault(prevColor, 0) - 1
                if (freq[prevColor] == 0)
                    freq.remove(prevColor)
                freq[cur] = freq.getOrDefault(cur, 0) + 1
            }
        } else {
            colors[pos] = cur
            freq[cur] = freq.getOrDefault(cur, 0) + 1
        }

        ans[i] = freq.size
    }

    return ans
}

fun occurrencesOfElement(nums: IntArray, queries: IntArray, x: Int): IntArray {
    if (!nums.contains(x)) {
        return IntArray(queries.size) { -1 }
    }

    val freq = hashMapOf<Int, Int>()
    var c = 0
    for (i in 0 until nums.size) {
        if (nums[i] == x) {
            c++
            freq[c] = i
        }
    }

    val ans = IntArray(queries.size) { -1 }
    for (i in 0 until queries.size) {
        if (freq.contains(queries[i])) {
            ans[i] = freq[queries[i]] ?: -1
        }
    }

    return ans
}

fun duplicateNumbersXOR(nums: IntArray): Int {
    val freq = hashMapOf<Int, Int>()
    nums.forEach {
        freq[it] = freq.getOrDefault(it, 0) + 1
    }

    var xor = 0

    for ((key, value) in freq) {
        if (value == 2)
            xor = xor xor key
    }

    return xor
}

fun waysToReachStair(k: Int): Int {
    val memo = hashMapOf<String, Int>()
    return dpExplore(1, 1, 0, k, memo)
}

fun dpExplore(at: Int, jump: Int, moveBack: Int, target: Int, memo: HashMap<String, Int>): Int {
    var c = 0
    if (at == target)
        c++

    val key = "$at|$jump|$moveBack"
    if (memo.containsKey(key))
        return memo[key]!!

    if (moveBack == 0 && at > 0) {
        c += dpExplore(at - 1, jump, 1, target, memo)
    }

    if (at + jump * 2 - 1 <= target + 1)
        c += dpExplore(at + jump, jump * 2, 0, target, memo)

    memo[key] = c

    return c
}

fun minOperationsToMakeMedianK(nums: IntArray, k: Int): Long {

    println(
        minOperationsToMakeMedianK(
            intArrayOf(
                2, 68, 15, 39, 30, 39, 97, 68
            ),
            2
        )
    )
    nums.sort()
    val mid = nums.size / 2

    println(nums.toList())

    if (nums.size % 2 == 1)
        return solveOdd(nums, k, mid)

    var largerIndex = 0
    if (nums[mid] > nums[mid - 1]) {
        largerIndex = mid
    } else
        largerIndex = mid - 1

    var plus = 0
    if (nums[mid] == nums[mid - 1]) {
        plus = Math.abs(nums[mid] - k)
    }

    return solveOdd(nums, k, largerIndex) + plus
}

private fun solveOdd(nums: IntArray, k: Int, mid: Int): Long {
    if (nums[mid] == k)
        return 0

    var operations = Math.abs(nums[mid] - k) + 0L

    for (i in 0 until mid) {
        val cur = nums[i]
        if (cur > k) {
            operations += cur - k
        }
    }

    for (i in mid + 1 until nums.size) {
        val cur = nums[i]
        if (cur < k) {
            operations += k - cur
        }
    }

    return operations
}

fun minimumSubstringsInPartition(s: String): Int {
    val memo = IntArray(s.length) { -1 }
    return dp(0, s, memo)
}

fun dp(at: Int, s: String, memo: IntArray): Int {
    if (at == s.length)
        return 0
    if (memo[at] != -1)
        return memo[at]

    val freq = hashMapOf<Char, Int>()
    val freqInt = hashMapOf<Int, Int>()

    var curMin = s.length
    for (i in at until s.length) {
        val cur = s[i]
        val prev = freq[cur]
        if (prev != null) {
            freqInt[prev] = (freqInt[prev] ?: 0) - 1
            if (freqInt[prev] == 0)
                freqInt.remove(prev)
            freqInt[prev + 1] = (freqInt[prev + 1] ?: 0) + 1
        } else {
            freqInt[1] = (freqInt[1] ?: 0) + 1
        }
        freq[cur] = freq.getOrDefault(cur, 0) + 1

        if (freqInt.size == 1) {
            val next = dp(i + 1, s, memo)
            curMin = Math.min(curMin, next + 1)
        }
    }

    memo[at] = curMin
    return curMin
}

fun maxScore(grid: List<List<Int>>): Int {
    var ans = -1

    val dp = Array(grid.size) { IntArray(grid[0].size) { Int.MAX_VALUE } }
    var min = Int.MAX_VALUE
    for (c in 0 until dp[0].size) {
        min = Math.min(grid[0][c], min)
        ans = Math.max(ans, grid[0][c] - min)
        dp[0][c] = min
    }

    min = Int.MAX_VALUE
    for (r in 0 until dp.size) {
        min = Math.min(grid[r][0], min)
        ans = Math.max(ans, grid[r][0] - min)
        dp[r][0] = min
    }

    for (i in 1 until grid.size) {
        for (j in 1 until grid[i].size) {
            var curMin = Math.min(dp[i - 1][j], dp[i][j - 1])
            ans = Math.max(ans, grid[i][j] - curMin)
            curMin = Math.min(curMin, grid[i][j])
            dp[i][j] = curMin
        }
    }

    return ans
}

fun maxPointsInsideSquare(points: Array<IntArray>, s: String): Int {

    if (s.toSet().size == s.length)
        return s.length

    val ps = mutableListOf<P>()
    for (i in 0 until points.size) {
        val p = points[i]
        val (x, y) = p
        val min = Math.min(Math.abs(x), Math.abs(y))
        ps.add(P(min, s[i]))
    }
    ps.sortWith(object : java.util.Comparator<P> {
        override fun compare(o1: P, o2: P): Int {
            return o1.min - o2.min
        }
    })

    val freq = BooleanArray(26) { false }
    var i = 0
    var j = 0
    var c = 0

    while (i < 100_001) {
        var l = 0
        while (j < ps.size) {
            val cur = ps[j]
            if (cur.min <= i) {
                if (freq[cur.tag - 'a']) {
                    return c - l
                } else {
                    freq[cur.tag - 'a'] = true
                    c++
                    l++
                }
                j++
            } else {
                break
            }
        }

        i++
    }

    return -1
}

class P(
    val min: Int,
    val tag: Char
)

fun shouldPop(curPoint: IntArray, k: Int, tag: Char, set: MutableSet<Char>): Boolean {
    val min = Math.min(Math.abs(curPoint[0]), Math.abs(curPoint[1]))
    return min <= k && !set.contains(tag)
}


fun satisfiesConditions(grid: Array<IntArray>): Boolean {

    for (i in 0 until grid.size) {
        for (j in 0 until grid[i].size) {
            if (i + 1 < grid.size && grid[i][j] != grid[i + 1][j])
                return false
            if (j + 1 < grid[i].size && grid[i][j] == grid[i][j + 1])
                return false
        }
    }

    return true
}

fun sumDigitDifferences(nums: IntArray): Long {

    val mapAtEach = Array(nums[0].toString().length) { hashMapOf<Int, Int>() }
    for (n in nums) {
        val nStr = n.toString()
        for (i in 0 until nStr.length) {
            val cur = nStr[i] - '0'
            val map = mapAtEach[i]
            map[cur] = map.getOrDefault(cur, 0) + 1
        }
    }

    var ans = 0L

    for (map in mapAtEach) {
        val freq = map.values.toList()
        for (i in 0 until freq.size) {
            for (j in i + 1 until freq.size) {
                ans += (freq[i] * freq[j])
            }
        }
    }

    return ans
}

fun isArraySpecial(nums: IntArray, queries: Array<IntArray>): BooleanArray {

    val treeSet = TreeSet<Int>()
    for (i in 0 until nums.size - 1) {
        if (nums[i] % 2 == nums[i + 1] % 2) {
            treeSet.add(i)
        }
    }
    val ans = BooleanArray(queries.size) { true }

    for (i in 0 until queries.size) {
        val (fr, to) = queries[i]
        if (fr != to) {
            val floor = treeSet.floor(to - 1) ?: -1
            if (floor >= fr)
                ans[i] = false
        }
    }

    return ans
}

fun solveMatrix(matrix: Array<IntArray>): Int {

    println(
        solveSuffix(
            intArrayOf(
                2, 3, 4, 10
            ),
            2
        )
    )

    println(
        solveMatrix(
            arrayOf(
                intArrayOf(10, 3, 10, 21, 2),
                intArrayOf(10, 1, 2, 3, 4),
                intArrayOf(20, 8, 7, 6, 5)
            )
        )
    )

    var ans = 0

    val dp = Array(matrix.size) { Array(matrix[0].size) { intArrayOf(Int.MAX_VALUE, 0) } }

    var min = Int.MAX_VALUE
    for (c in 0 until matrix[0].size) {
        min = Math.min(min, matrix[0][c])
        if (min < dp[0][c][0]) {
            dp[0][c][0] = min
            dp[0][c][1] = matrix[0][c] - min
            ans = Math.max(ans, dp[0][c][1])
        } else {
            dp[0][c][0] = min
        }
    }

    min = Int.MAX_VALUE
    for (r in 0 until matrix.size) {
        min = Math.min(min, matrix[r][0])
        if (min < dp[r][0][0]) {
            dp[r][0][0] = min
            dp[r][0][1] = matrix[r][0] - min
            ans = Math.max(ans, dp[r][0][1])
        } else {
            dp[r][0][0] = min
        }
    }

    for (r in 1 until matrix.size) {
        for (c in 1 until matrix[r].size) {
            val cur = matrix[r][c]
            if (cur > dp[r - 1][c][0]) {
                val score = cur - dp[r - 1][c][0] + dp[r - 1][c][1]
                ans = Math.max(ans, score)
            }
            if (cur > dp[r][c - 1][0]) {
                val score = cur - dp[r][c - 1][0] + dp[r][c - 1][1]
                ans = Math.max(ans, score)
            }
        }
    }

    return ans
}

fun solveSuffix(nums: IntArray, k: Int): Int {

    var ans = Int.MIN_VALUE
    val suffix = IntArray(nums.size) { 0 }
    for (i in nums.size - 1 downTo 0) {
        val mod = i % k
        suffix[mod] += nums[i]
        ans = Math.max(ans, suffix[mod])
    }

    return ans
}

fun maximumEnergy(energy: IntArray, k: Int): Int {

    var ans = Int.MIN_VALUE

    for (e in energy) {
        ans = Math.max(ans, e)
    }

    if (ans < 0)
        return ans

    for (i in 0 until k) {
        var at = i
        var sum = 0
        while (at < energy.size) {
            sum += energy[at]
            sum = Math.max(sum, 0)
            at += k
        }
        ans = Math.max(ans, sum)
    }

    return ans
}

fun findPermutationDifference(s: String, t: String): Int {

    val p1 = hashMapOf<Char, Int>()
    val p2 = hashMapOf<Char, Int>()

    for (i in 0 until s.length) {
        p1[s[i]] = i
    }

    for (j in 0 until t.length) {
        p2[t[j]] = j
    }

    var sum = 0
    for ((char, pos) in p1) {
        val pos2 = p2[char] ?: 0
        sum += Math.abs(pos2 - pos)
    }

    return sum
}


fun minAnagramLength(s: String): Int {

    for (i in 2..s.length / 2) {
        if (s.length % i == 0) {
            val freq = IntArray(26) { 0 }
            for (j in 0 until i) {
                freq[s[j] - 'a']++
            }
            println(freq.toList())
            var valid = true

            for (j in i until s.length step i) {
                val newFreq = IntArray(26) { 0 }
                for (k in j until j + i) {
                    newFreq[s[k] - 'a']++
                }

                for (k in 0 until 26) {
                    if (freq[k] != newFreq[k]) {
                        valid = false
                        break
                    }
                }
            }
            if (valid)
                return i
        }
    }

    return s.length
}

fun minimumOperationsToMakeKPeriodic(word: String, k: Int): Int {

    val map = hashMapOf<String, Int>()
    for (i in 0 until word.length step k) {
        val sub = word.substring(i, i + k)
        map[sub] = map.getOrDefault(sub, 0) + 1
    }

    val list = map.values.sortedWith(Collections.reverseOrder())
    val max = list[0]

    var sum = 0
    for (i in 1 until list.size)
        sum += list[i]

    return sum
}

fun isValid(word: String): Boolean {
    if (word.length < 3) return false
    val filter = word.toCharArray().filter { it.isLetterOrDigit() }
    if (filter.isNotEmpty()) return false
    val vowels = listOf('a', 'e', 'i', 'o', 'u')
    var valid1 = false
    var valid2 = false
    for (w in word) {
        if (vowels.contains(w.lowercaseChar())) {
            valid1 = true
        } else
            valid2 = true

    }

    return valid1 && valid2
}

fun minEnd(n: Int, x: Int): Long {
    println(
        minEnd(
            2, 7
        )
    )
    return 0L
}

fun minimumAddedInteger(nums1: IntArray, nums2: IntArray): Int {

    println(
        minimumAddedInteger(
            intArrayOf(4, 20, 16, 12, 8),
            intArrayOf(14, 18, 10)
        )
    )
    val target = nums2.sorted().toIntArray()
    var ans = Int.MAX_VALUE

    for (i in 0 until nums1.size) {
        for (j in i + 1 until nums1.size) {
            val list = mutableListOf<Int>()
            for (k in 0 until nums1.size) {
                if (k != i && k != j) {
                    list.add(nums1[k])
                }
            }

            list.sort()
            val dif = addedInteger(list.toIntArray(), target)
            var valid = true
            for (k in 0 until list.size) {
                if (list[k] + dif != target[k]) {
                    valid = false
                    break
                }
            }
            if (valid) {
                ans = Math.min(ans, dif)
            }
        }
    }

    return ans
}

fun addedInteger(nums1: IntArray, nums2: IntArray): Int {

    if (nums1[0] >= nums2[0])
        return nums1[0] - nums2[0]

    return nums2[0] - nums1[0]
}


data class Leetcode(val name: String, val surName: String)

@JvmInline
value class Name(val value: String)

@JvmInline
value class SurName(val value: String)

fun minimumCost(n: Int, edges: Array<IntArray>, query: Array<IntArray>): IntArray {

    val adj = Array(n) { mutableListOf<IntArray>() }
    for ((a, b, w) in edges) {
        adj[a].add(intArrayOf(b, w))
        adj[b].add(intArrayOf(a, w))
    }

    val groups = IntArray(n) { -1 }
    val minGrouping = hashMapOf<Int, Int>()
    var group = 1

    for (i in 0 until n) {
        if (groups[i] == -1) {
            groups[i] = group
            val weighs = mutableListOf<Int>()
            val nodes = mutableSetOf<Int>()
            dfs(group, i, groups, adj, weighs, nodes)
            group++
            var and = weighs[0]

            for (j in 1 until weighs.size) {
                and = and and weighs[j]
            }

            for (k in nodes) {
                minGrouping[k] = and
            }
        }
    }

    val ans = IntArray(query.size) { -1 }
    for (i in 0 until query.size) {
        val (a, b) = query[i]
        if (groups[a] == groups[b]) {
            ans[i] = minGrouping[groups[a]] ?: -1
        }
    }

    return ans
}

private fun dfs(
    group: Int,
    at: Int,
    groups: IntArray,
    adj: Array<MutableList<IntArray>>,
    weighs: MutableList<Int>,
    nodes: MutableSet<Int>
) {
    nodes.add(at)
    groups[at] = group
    for (neigh in adj[at]) {
        if (groups[neigh[0]] == -1) {
            weighs.add(neigh[1])
            dfs(group, neigh[0], groups, adj, weighs, nodes)
        }
    }
}

fun minimumTime(n: Int, edges: Array<IntArray>, disappear: IntArray): IntArray {

    val adj = hashMapOf<Int, List<DestNode>>()

    for ((a, b, w) in edges) {
        adj[a] = adj.getOrDefault(a, listOf()) + DestNode(b, w)
        adj[b] = adj.getOrDefault(b, listOf()) + DestNode(a, w)
    }

    //curNode,curTime
    val queue = PriorityQueue(object : Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            return o1[1] - o2[1]
        }
    })
    queue.add(intArrayOf(0, 0))
    val ans = IntArray(n) { -1 }

    while (queue.isNotEmpty()) {
        val poll = queue.poll()
        if (ans[poll[0]] < poll[1] && poll[1] < disappear[poll[0]]) {
            ans[poll[0]] = poll[1]
            for (neigh in adj[poll[0]] ?: continue) {
                queue.add(intArrayOf(neigh.dest, poll[1] + neigh.weight))
            }
        }
    }

    println(ans.toList())

    return ans
}

class DestNode(val dest: Int, val weight: Int)

fun minRectanglesToCoverPoints(points: Array<IntArray>, w: Int): Int {

    val pointsByX = mutableListOf<Int>()

    for ((x, y) in points) {
        pointsByX.add(x)
    }

    pointsByX.sort()

    var at = 0
    var c = 0

    while (at < points.size) {
        val x = pointsByX[at]
        val max = x + w
        c++
        while (at < pointsByX.size && pointsByX[at] <= max) {
            at++
        }
    }

    println(
        minRectanglesToCoverPoints(
            arrayOf(
                intArrayOf(0, 0),
                intArrayOf(1, 1),
                intArrayOf(2, 2),
                intArrayOf(3, 3)
            ),
            1
        )
    )

    return c
}

fun getSmallestString(s: String, k: Int): String {

    val ans = StringBuilder(s)
    var dif = k

    for (i in 0 until s.length) {
        var cur = 'a'
        while (getMinDistance(s[i], cur) > dif) {
            cur = cur.inc()
        }
        val dist = getMinDistance(ans[i], cur)
        dif -= getMinDistance(ans[i], cur)
        ans[i] = cur
    }

    return ans.toString()
}

fun getMinDistance(a: Char, b: Char): Int {
    val right = Math.abs(a - b)
    val l = 26 - right
    return Math.min(right, l)
}

fun longestMonotonicSubarray(nums: IntArray): Int {

    var inc = 1
    var max = 1
    for (i in 1 until nums.size) {
        if (nums[i] > nums[i - 1]) {
            inc++
        } else {
            inc = 1
        }
        max = Math.max(max, inc)
    }

    for (i in 1 until nums.size) {
        if (nums[i] < nums[i - 1]) {
            inc++
        } else {
            inc = 1
        }
        max = Math.max(max, inc)
    }

    return max
}


//3
//0,1,2,3,4,5,6

fun sumOfPowers(nums: IntArray, k: Int): Int {

    nums.sort()
    var ans = 0
    for (i in 0..nums.size - k) {
        var minDif = Int.MAX_VALUE
        //0 till 0+3

        for (j in i + 1 until i + k - 1) {
            minDif = Math.min(minDif, nums[j] - nums[j - 1])
        }

        for (j in i + k - 1 until nums.size) {
            println("Prev : ${nums[i + k - 2]} Cur : ${nums[j]}")

            val min = Math.min(minDif, nums[j] - nums[i + k - 2])
            println("Min : $min")
            ans += min
        }
    }

    return ans
}

fun minimumSubarrayLength(nums: IntArray, k: Int): Int {

    if (k == 0)
        return 1

    var ans = Int.MAX_VALUE
    val freq = IntArray(32) { 0 }
    var r = 0
    for (l in 0 until nums.size) {
        while (r < nums.size && isNotEnough(freq, k)) {
            for (i in 0 until 32) {
                if (nums[r] and (1 shl i) != 0) {
                    freq[i]++
                }
            }
            r++
        }

        if (!isNotEnough(freq, k)) {
            val size = r - l
            ans = Math.min(ans, size)
        }

        for (i in 0 until 32) {
            if (nums[l] and (1 shl i) != 0) {
                freq[i]--
            }
        }
    }

    if (ans != Int.MAX_VALUE)
        return ans

    return -1
}


fun isNotEnough(freq: IntArray, k: Int): Boolean {

    var pow = 0
    for (i in 0 until freq.size) {
        if (freq[i] > 0) {
            pow = pow xor (1 shl i)
        }
    }

    return pow < k
}

fun countAlternatingSubarrays(nums: IntArray): Long {
    var ans = 0L
    var alter = 1
    for (i in 1 until nums.size) {
        if (nums[i] != nums[i - 1]) {
            alter++
        } else {
            ans += (alter * (alter + 1L)) / 2
            alter = 1
        }
    }

    ans += (alter * (alter + 1L)) / 2
    return ans
}

fun unmarkedSumArray(nums: IntArray, queries: Array<IntArray>): LongArray {


    return longArrayOf()
}

fun minimumDeletions(word: String, k: Int): Int {

    val freq = hashMapOf<Char, Int>()
    for (w in word) {
        freq[w] = freq.getOrDefault(w, 0) + 1
    }

    val list = freq.values.toList().sorted()
    var ans = Int.MAX_VALUE

    var start = 0
    for (i in 0 until list.size) {
        var c = start
        for (j in i + 1 until list.size) {
            if (list[j] > list[i] + k) {
                c += (list[j] - (list[i] + k))
            }
        }
        ans = Math.min(ans, c)
        start += list[i]
    }

    return ans
}

fun countSubstrings(s: String, c: Char): Long {

    var freq = 0L
    for (i in 0 until s.length) {
        if (s[i] == c) {
            freq++
        }
    }

    return (freq * (freq * 2)) / 2
}

fun shortestSubstrings(arr: Array<String>): Array<String> {

    val ansArray = Array(arr.size) { "" }

    for (i in 0 until arr.size) {
        val str = arr[i]
        for (size in 1..str.length) {
            val list = mutableListOf<String>()
            for (j in 0..str.length - size) {
                val sub = str.substring(j, j + size)
                var valid = true
                for (k in 0 until arr.size) {
                    if (k != i) {
                        val other = arr[k]
                        if (other.contains(sub)) {
                            valid = false
                        }
                    }
                }
                if (valid) {
                    list.add(sub)
                }
            }
            if (list.isNotEmpty()) {
                list.sort()
                ansArray[i] = list[0]
                break
            }
        }
    }

    return ansArray
}

fun numMusicPlaylists(n: Int, goal: Int, k: Int): Int {
    println(
        numMusicPlaylists(
            3, 3, 1
        )
    )

    return 0
}

fun countOfPairs(n: Int, x: Int, y: Int): IntArray {

    val ans = IntArray(n) { 0 }

    for (i in 1..n) {
        val visited = BooleanArray(n + 1) { false }
        visited[i] = true
        val queue = LinkedList<IntArray>()
        queue.add(intArrayOf(i, 0))
        while (queue.isNotEmpty()) {
            val (cur, dist) = queue.poll()
            if (cur == x && !visited[y]) {
                visited[y] = true
                ans[dist]++
                queue.add(
                    intArrayOf(
                        y,
                        dist + 1
                    )
                )
            }
            if (cur == y && !visited[x]) {
                visited[x] = true
                ans[dist]++
                queue.add(
                    intArrayOf(
                        x,
                        dist + 1
                    )
                )
            }
            if (cur + 1 <= n && !visited[cur + 1]) {
                visited[cur + 1] = true
                ans[dist]++
                queue.add(
                    intArrayOf(
                        cur + 1,
                        dist + 1
                    )
                )
            }
            if (cur - 1 >= 1 && !visited[cur - 1]) {
                visited[cur - 1] = true
                ans[dist]++
                queue.add(
                    intArrayOf(
                        cur - 1,
                        dist + 1
                    )
                )
            }
        }
    }

    return ans
}

fun countPairsOfConnectableServers(edges: Array<IntArray>, signalSpeed: Int): IntArray {

    val pairs = Array(edges.size + 1) { listOf<Int>() }
    val starting = mutableListOf<Int>()
    val adj = Array(edges.size + 1) { mutableListOf<List<Int>>() }

    for ((a, b, w) in edges) {
        adj[a].add(listOf(b, w))
        adj[b].add(listOf(a, w))
    }

    for (i in 0 until adj.size) {
        if (adj[i].size == 1) {
            starting.add(adj[i][0][0])
        }
    }

    println("Starting : $starting")
    dfs(starting[0], -1, adj, pairs)


    return intArrayOf()
}

fun dfs(at: Int, parent: Int, adj: Array<MutableList<List<Int>>>, pairs: Array<List<Int>>): IntArray {

    var sum = 0
    var c = 0



    return intArrayOf()
}

fun makeSimilar(nums: IntArray, target: IntArray): Long {

    nums.sort()
    target.sort()
    var ans = 0L
    var l = 0
    var r = nums.size - 1
    while (l < r) {
        val lDif = Math.abs(target[l] - nums[l])
        val rDif = Math.abs(target[r] - nums[r])
        if (lDif == 0) {
            l++
        } else if (rDif == 0) {
            r--
        } else {
            if (lDif == rDif) {
                val oper = lDif / 2
                ans += oper
                l++
                r--
            } else if (lDif > rDif) {
                nums[l] += rDif
                r--
                ans += (rDif / 2)
            } else {
                l++
                nums[r] -= lDif
                ans += (lDif / 2)
            }
        }
    }

    return ans
}

fun sumImbalanceNumbers(nums: IntArray): Int {

    var ans = 0
    for (i in 0 until nums.size) {
        val treeSet = TreeSet(listOf(nums[i]))
        var c = 0
        for (j in i + 1 until nums.size) {
            val cur = nums[j]
            if (cur < treeSet.first()) {
                val dif = treeSet.first() - cur
                if (dif > 1) {
                    c++
                }
            } else if (cur > treeSet.last()) {
                val dif = cur - treeSet.last()
                if (dif > 1) {
                    c++
                }
            } else if (!treeSet.contains(cur)) {
                val floor = treeSet.floor(cur - 1)!!
                val ceil = treeSet.ceiling(cur + 1)!!
                val abs1 = cur - floor
                val abs2 = ceil - cur
                if (abs1 > 1 && abs2 > 1) {
                    c++
                } else if (abs1 == 1 && abs2 == 1) {
                    c--
                }
            }
            treeSet.add(cur)
            ans += c
        }
    }

    return ans
}

private val mod = Math.pow(10.0, 9.0).toInt() + 5

fun beautifulIndices(s: String, a: String, b: String, k: Int): List<Int> {

    val l1 =
        rollingHashIndices(
            s,
            a,
            26,
            'a'
        )

    val treeSet = TreeSet(
        rollingHashIndices(
            s, b,
            26,
            'a'
        )
    )

    val ans = mutableListOf<Int>()

    for (n in l1) {
        val min = treeSet.floor(n)
        val max = treeSet.ceiling(n)
        var inRange = false
        if (min != null) {
            if (n - min <= k)
                inRange = true
        }
        if (max != null) {
            if (max - n <= k)
                inRange = true
        }
        if (inRange)
            ans.add(n)
    }

    return ans
}

private fun rollingHashIndices(s: String, t: String, multi: Int, remove: Char): List<Int> {

    val list = mutableListOf<Int>()
    var targetHash = 0L
    var sHash = 0L
    var pow = 1L
    var last = 0L
    for (i in t.length - 1 downTo 0) {
        val cur = t[i] - remove + 1
        sHash += (s[i] - remove + 1) * pow
        targetHash += (cur * pow)
        last = pow
        pow *= multi
        targetHash %= mod
        sHash %= mod
        pow %= mod
    }

    if (sHash == targetHash) {
        if (s.substring(0, t.length) == t) {
            list.add(0)
        }
    }

    for (i in t.length until s.length) {
        val left = ((s[i - t.length] - remove) + 1) * last
        val cur = ((s[i] - remove) + 1)
        sHash -= left
        sHash *= 26
        while (sHash < 0)
            sHash += mod
        sHash += cur
        sHash %= mod
        if (sHash == targetHash) {
            val sub = s.substring(i - t.length + 1, i + 1)
            if (sub == t)
                list.add(i - t.length + 1)
        }
    }

    return list
}

fun resultArray(nums: IntArray): IntArray {

    val billion = Math.pow(10.0, 9.0).toInt()
    val tree1 = DynamicSegmentTree(billion)
    val tree2 = DynamicSegmentTree(billion)

    val arr1 = mutableListOf(nums[0])
    val arr2 = mutableListOf(nums[1])

    tree1.add(nums[0])
    tree2.add(nums[1])

    for (i in 2 until nums.size) {
        val cur = nums[i]
        val first = tree1.findCountInRange(cur + 1, billion)
        val second = tree2.findCountInRange(cur + 1, billion)
        if (first > second) {
            tree1.add(cur)
            arr1.add(cur)
        } else if (first < second) {
            tree2.add(cur)
            arr2.add(cur)
        } else {
            if (arr1.size == arr2.size) {
                arr1.add(cur)
                tree1.add(cur)
            } else if (arr1.size < arr2.size) {
                arr1.add(cur)
                tree1.add(cur)
            } else {
                arr2.add(cur)
                tree2.add(cur)
            }
        }
    }

    return (arr1 + arr2).toIntArray()
}