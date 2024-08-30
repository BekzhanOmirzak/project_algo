package utils.sparsetable

import kotlin.math.log2

fun main() {

    println(
        countSubarrays(
            intArrayOf(1, 1, 2),
            1
        )
    )

}

fun countSubarrays(nums: IntArray, k: Int): Long {

    val obj = SparseTable(nums)
    var ans = 0L
    for (l in 0 until nums.size) {
        val right = findRightMostIndex(l, obj, k, nums.size)
        if (right != -1) {
            val left = findLeftMostIndex(l, right, obj, k)
            ans += (right - left + 1)
        }
    }

    return ans
}

private fun findLeftMostIndex(left: Int, right: Int, obj: SparseTable, k: Int): Int {

    var l = left
    var r = right
    var ans = right
    while (l <= r) {
        val mid = (r + l) / 2
        val and = obj.findRangeAnd(left, mid)
        if (and == k) {
            ans = Math.min(ans, mid)
            r = mid - 1
        } else if (and > k) {
            l = mid + 1
        } else {
            r = mid - 1
        }
    }

    return ans
}

private fun findRightMostIndex(left: Int, obj: SparseTable, k: Int, n: Int): Int {

    var l = left
    var r = n - 1
    var ans = -1
    while (l <= r) {
        val mid = (r + l) / 2
        val and = obj.findRangeAnd(left, mid)
        if (and >= k) {
            if (and == k)
                ans = Math.max(ans, mid)
            l = mid + 1
        } else {
            r = mid - 1
        }
    }

    return ans
}

class SparseTable(nums: IntArray) {

    private val maxPower = largestPowerOfTwoWithinRange(nums.size) + 1
    private val lookUp = Array(nums.size) { IntArray(maxPower) }

    init {
        for (i in 0 until nums.size) {
            lookUp[i][0] = nums[i]
        }

        for (j in 1 until maxPower) {
            for (i in 0 until nums.size) {
                val nextIndex = i + (1 shl (j - 1))
                if (nextIndex < nums.size) {
                    lookUp[i][j] = lookUp[i][j - 1] + lookUp[nextIndex][j - 1]
                } else {
                    lookUp[i][j] = lookUp[i][j - 1]
                }
            }
        }
    }

    fun findRangeAnd(l: Int, r: Int): Int {
        var l = l
        var res = lookUp[l][0]
        for (j in maxPower - 1 downTo 0) {
            if ((1 shl j) <= r - l + 1) {
                res = res and lookUp[l][j]
                l += 1 shl j
            }
        }
        return res
    }

    fun findRangeSum(l: Int, r: Int): Int {

        var l = l
        var res = 0
        for (j in maxPower - 1 downTo 0) {
            if ((1 shl j) <= r - l + 1) {
                res += lookUp[l][j]
                l += 1 shl j
            }
        }

        return res
    }

    private fun findRangeSumSecond(l: Int, r: Int, tableSum: Array<IntArray>): Int {
        var sum = 0
        var left = l
        var power: Int

        while (left <= r) {
            power = largestPowerOfTwoWithinRange(r - left + 1)
            sum += tableSum[left][power]
            left += 1 shl power
        }

        return sum
    }

    private fun largestPowerOfTwoWithinRange(range: Int): Int {
        return log2(range.toDouble()).toInt()
    }

}

