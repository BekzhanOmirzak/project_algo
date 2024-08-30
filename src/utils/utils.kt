package utils

fun main() {


}

val mod = Math.pow(10.0, 9.0).toInt() + 5

private val dirs = arrayOf(
    intArrayOf(0, 1), intArrayOf(0, -1), intArrayOf(1, 0), intArrayOf(-1, 0)
)

private fun readInt() = readString().toInt()
private fun readString() = readLine().toString()
private fun readDouble() = readString().toDouble()
private fun readListString() = readString().split(" ")
private fun readListInt() = readString().split(" ").map { it.toInt() }

private fun convertStrToList(str: String) = str.split(" ").map { it.toInt() }

private fun <T> List<T>.getFreq(): HashMap<T, Int> {
    val freq = hashMapOf<T, Int>()
    for (l in this) {
        freq[l] = freq.getOrDefault(l, 0) + 1
    }
    return freq
}

private fun <T> Array<T>.permute(result: (Array<T>) -> Unit) {
    permuteHelper(0, this, result)
}

private fun <T> permuteHelper(at: Int, arr: Array<T>, result: (Array<T>) -> Unit) {
    if (at == arr.size) {
        result(arr)
    }
    for (i in at until arr.size) {
        swap(i, at, arr)
        permuteHelper(at + 1, arr, result)
        swap(i, at, arr)
    }
}

private fun <T> swap(i: Int, j: Int, nums: Array<T>) {
    val temp = nums[i]
    nums[i] = nums[j]
    nums[j] = temp
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

private fun getSumPre(l: Int, r: Int, prefix: IntArray): Int {
    if (l == 0)
        return prefix[r]
    return prefix[r] - prefix[l - 1]
}

private fun isPrime(n: Int): Boolean {
    if (n <= 1)
        return false
    if (n == 2)
        return true
    if (n % 2 == 0)
        return false

    val sqrt = Math.sqrt(n.toDouble()).toInt()
    for (d in 3..sqrt) {
        if (n % d == 0)
            return false
    }
    return true
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
        res %= mod
    }
    return res
}


//TC : O(each) to O(log(each))
private fun modExp(base: Long, exp: Long, mod: Int): Long {
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


















