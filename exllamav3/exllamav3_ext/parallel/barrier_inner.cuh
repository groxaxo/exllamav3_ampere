
__device__ __forceinline__ void pg_barrier_inner
(
    PGContext* __restrict__ ctx,
    uint32_t device_mask,
    int this_device,
    int coordinator_device,
    uint32_t* abort_flag
)
{
    if (!blockIdx.x && !blockIdx.y && !blockIdx.z && !threadIdx.x && !threadIdx.y && !threadIdx.z)
    {
        uint32_t* epoch_ptr     = &ctx->barrier_epoch;
        uint32_t* epoch_dev_ptr = ctx->barrier_epoch_device;

        // Snapshot current epoch with acquire semantics for cross-GPU visibility
        const uint32_t epoch = ldg_acquire_sys_u32(epoch_ptr);

        // Publish arrival with release semantics so other GPUs see this write
        stg_release_sys_u32(&epoch_dev_ptr[this_device], epoch);

        if (this_device == coordinator_device)
        {
            uint64_t deadline = sync_deadline();
            uint32_t pending = device_mask & ~(1 << this_device);

            // Wait for other participants to arrive at epoch, using acquire-sys reads
            // for cross-GPU visibility via PCIe-attached pinned host memory.
            uint32_t sleep = SYNC_MIN_SLEEP;
            while (pending)
            {
                uint32_t pending_t = pending;
                for (int i = 0; i < MAX_DEVICES; ++i)
                {
                    if (!(pending & (1 << i))) continue;
                    if (ldg_acquire_sys_u32(&epoch_dev_ptr[i]) == epoch)
                        pending &= ~(1 << i);
                }

                if (pending == pending_t)
                {
                    __nanosleep(sleep);
                    if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                    else *abort_flag = check_timeout(ctx, deadline, "barrier");
                    if (*abort_flag) break;
                }
                else sleep = SYNC_MIN_SLEEP;
            }

            // Release: bump epoch with release semantics
            stg_release_sys_u32(epoch_ptr, epoch + 1);
        }
        else
        {
            uint64_t deadline = sync_deadline();

            // Wait for coordinator to bump epoch using acquire-sys reads
            uint64_t sleep = SYNC_MIN_SLEEP;
            while (ldg_acquire_sys_u32(epoch_ptr) == epoch)
            {
                __nanosleep(sleep);
                if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                else *abort_flag = check_timeout(ctx, deadline, "barrier");
                if (*abort_flag) break;
            }
        }
    }
    __syncthreads();
}