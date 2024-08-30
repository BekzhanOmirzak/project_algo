package test

import java.util.Comparator
import java.util.PriorityQueue
import java.util.Stack
import kotlin.random.Random

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

fun findKthSmallest(coins: IntArray, k: Int): Long {

    coins.sort()
    var l = 0L
    var r = Long.MAX_VALUE
    var ans = Long.MAX_VALUE

    while (l <= r) {
        val mid = l + (r - l) / 2L
        val many = bs(mid, coins, k)
        if (many >= k) {
            ans = Math.min(ans, mid)
            r = mid - 1
        } else
            l = mid + 1
    }

    return ans
}

fun bs(smallest: Long, coins: IntArray, k: Int): Int {

    var c = 0

    var n = 0L
    val sum = LongArray(coins.size) { 0L }

    while (true) {
        val set = mutableSetOf<Long>()
        n += coins[0]
        if (n <= smallest)
            set.add(n)
        for (i in 1 until coins.size) {
            val next = sum[i] + coins[i]
            if (next <= n && next <= smallest) {
                sum[i] += coins[i].toLong()
                set.add(next)
            }
        }

        c += set.size
        if (c >= k || set.isEmpty())
            break

    }

    return c
}

class NNode(val num: Long, val coin: Int)

fun findLatestTime(s: String): String {

    val (hour, minute) = s.split(":")
    val hourBuilder = StringBuilder(hour)
    if (hourBuilder[0] == '?') {
        if (hourBuilder[1] == '?') {
            hourBuilder[0] = '1'
            hourBuilder[1] = '1'
        } else {
            if (hourBuilder[1] - '0' in 2..9) {
                hourBuilder[1] = '0'
            } else
                hourBuilder[1] = '1'
        }
    }

    if (hourBuilder[1] == '?') {
        if (hourBuilder[0] == '1')
            hourBuilder[1] = '1'
        else {
            hourBuilder[1] = '9'
        }
    }

    val minuteBuilder = StringBuilder(minute)

    if (minuteBuilder[0] == '?')
        minuteBuilder[0] = '5'

    if (minuteBuilder[1] == '?')
        minuteBuilder[1] = '9'

    return "$hourBuilder:$minuteBuilder"
}


class FirstThread : Thread() {

    override fun run() {
        var c = 0
        while (true) {
            println("Cur Counter : $c")
            c++
            sleep(1000L)
        }
    }

}
