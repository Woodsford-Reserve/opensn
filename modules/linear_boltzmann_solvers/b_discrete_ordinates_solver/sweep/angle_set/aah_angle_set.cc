#include "modules/linear_boltzmann_solvers/b_discrete_ordinates_solver/sweep/angle_set/aah_angle_set.h"
#include "modules/linear_boltzmann_solvers/b_discrete_ordinates_solver/sweep_chunks/sweep_chunk.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"

namespace opensn
{
namespace lbs
{

AAH_AngleSet::AAH_AngleSet(size_t id,
                           size_t num_groups,
                           size_t group_subset,
                           const SPDS& spds,
                           std::shared_ptr<FLUDS>& fluds,
                           std::vector<size_t>& angle_indices,
                           std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                           int sweep_eager_limit,
                           const MPICommunicatorSet& comm_set)
  : AngleSet(id, num_groups, spds, fluds, angle_indices, boundaries, group_subset),
    async_comm_(*fluds, num_groups_, angle_indices.size(), sweep_eager_limit, comm_set)
{
}

void
AAH_AngleSet::InitializeDelayedUpstreamData()
{
  async_comm_.InitializeDelayedUpstreamData();
}

AngleSetStatus
AAH_AngleSet::AngleSetAdvance(SweepChunk& sweep_chunk,
                              const std::vector<size_t>& timing_tags,
                              AngleSetStatus permission)
{
  if (executed_)
  {
    if (not async_comm_.DoneSending())
      async_comm_.ClearDownstreamBuffers();
    return AngleSetStatus::FINISHED;
  }

  // Check upstream data available
  AngleSetStatus status = async_comm_.ReceiveUpstreamPsi(static_cast<int>(this->GetID()));

  // Also check boundaries
  for (auto& [bid, bndry] : boundaries_)
    if (not bndry->CheckAnglesReadyStatus(angles_, group_subset_))
    {
      status = AngleSetStatus::RECEIVING;
      break;
    }

  if (status == AngleSetStatus::RECEIVING)
    return status;
  else if (status == AngleSetStatus::READY_TO_EXECUTE and permission == AngleSetStatus::EXECUTE)
  {
    async_comm_.InitializeLocalAndDownstreamBuffers();

    log.LogEvent(timing_tags[0], Logger::EventType::EVENT_BEGIN);
    sweep_chunk.Sweep(*this); // Execute chunk
    log.LogEvent(timing_tags[0], Logger::EventType::EVENT_END);

    // Send outgoing psi and clear local and receive buffers
    async_comm_.SendDownstreamPsi(static_cast<int>(this->GetID()));
    async_comm_.ClearLocalAndReceiveBuffers();

    // Update boundary readiness
    for (auto& [bid, bndry] : boundaries_)
      bndry->UpdateAnglesReadyStatus(angles_, group_subset_);

    executed_ = true;
    return AngleSetStatus::FINISHED;
  }
  else
    return AngleSetStatus::READY_TO_EXECUTE;
}

AngleSetStatus
AAH_AngleSet::FlushSendBuffers()
{
  if (not async_comm_.DoneSending())
    async_comm_.ClearDownstreamBuffers();

  if (async_comm_.DoneSending())
    return AngleSetStatus::MESSAGES_SENT;

  return AngleSetStatus::MESSAGES_PENDING;
}

int
AAH_AngleSet::GetMaxBufferMessages() const
{
  return async_comm_.max_num_mess;
}

void
AAH_AngleSet::SetMaxBufferMessages(int new_max)
{
  async_comm_.max_num_mess = new_max;
}

void
AAH_AngleSet::ResetSweepBuffers()
{
  async_comm_.Reset();
  executed_ = false;
}

bool
AAH_AngleSet::ReceiveDelayedData()
{
  return async_comm_.ReceiveDelayedData(static_cast<int>(this->GetID()));
}

const double*
AAH_AngleSet::PsiBoundary(uint64_t bndry_map,
                          unsigned int angle_num,
                          uint64_t cell_local_id,
                          unsigned int face_num,
                          unsigned int fi,
                          int g,
                          size_t gs_ss_begin,
                          bool surface_source_active)
{
  if (boundaries_[bndry_map]->IsReflecting())
    return boundaries_[bndry_map]->HeterogeneousPsiIncoming(
      cell_local_id, face_num, fi, angle_num, g, gs_ss_begin);

  if (not surface_source_active)
    return boundaries_[bndry_map]->ZeroFlux(g);

  return boundaries_[bndry_map]->HeterogeneousPsiIncoming(
    cell_local_id, face_num, fi, angle_num, g, gs_ss_begin);
}

double*
AAH_AngleSet::ReflectingPsiOutboundBoundary(uint64_t bndry_map,
                                            unsigned int angle_num,
                                            uint64_t cell_local_id,
                                            unsigned int face_num,
                                            unsigned int fi,
                                            size_t gs_ss_begin)
{
  return boundaries_[bndry_map]->HeterogeneousPsiOutgoing(
    cell_local_id, face_num, fi, angle_num, gs_ss_begin);
}

} // namespace lbs
} // namespace opensn
